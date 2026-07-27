"""The one place anything in this project reaches a model.

Every model call — the task analyzer, the dataset generator, the judge, and every
call a candidate workflow makes — goes through `ModelClient.call`. That single
chokepoint is what makes cost measurable no matter what a workflow's code does.

All calls go through OpenRouter (one API over every provider), so any OpenRouter
model id works and the bill lands in one place. OpenRouter reports what it
actually charged per call; that billed figure is preferred over local price math
wherever it is available.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
import openai
from pydantic import BaseModel

from .models import ModelCatalog

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Server-side tools, model-callable through OpenRouter: the model decides when
# and how often to use each, OpenRouter executes it and bills per use.
#   web_search — agentic search; native provider search when the model has it,
#                Exa otherwise (~$0.005/request)
#   web_fetch  — read a URL already present in the conversation
#   subagent   — delegate a self-contained subtask to a worker model
#                mid-generation (the worker defaults to the calling model)
# "web" is accepted as an alias from the deprecated one-plugin era. There is
# still no code execution: OpenRouter runs no code server-side.
TOOL_DEFS = {
    "web_search": {"type": "openrouter:web_search"},
    "web_fetch": {"type": "openrouter:web_fetch"},
    "subagent": {"type": "openrouter:subagent"},
}
TOOL_ALIASES = {"web": "web_search"}

# Request plugins — NOT model-callable tools: each runs once per request when
# enabled, mutating it. Applied to every call via `call.plugins` in config.
#   response-healing    — auto-repair malformed JSON output
#   context-compression — middle-out truncation when a prompt exceeds context
PLUGIN_DEFS = {
    "response-healing": {"id": "response-healing"},
    "context-compression": {"id": "context-compression"},
}


@dataclass
class ApiResponse:
    """The result of one completed call to the model API.

    Attributes:
        text: The final answer text.
        blocks: Ancillary content the call produced — for web-plugin calls, the
            URL citations. Kept for the trace.
        usage: Token counts with keys "input", "output", "cache_write" and
            "cache_read".
        truncated: True when the reply was cut off at the output ceiling —
            `text` is then a partial answer, not a finished one.
        cost: What OpenRouter actually billed for the call, in USD, when it
            reported it. None falls back to catalog price math.
    """
    text: str
    blocks: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    truncated: bool = False
    cost: Optional[float] = None


class ModelClient:
    """A thin, cache-aware wrapper over OpenRouter's chat completions API.

    Holds the model catalog and per-call settings so callers pass neither around.

    Attributes:
        catalog: The models this client may call, and their prices.
        cfg: A `CallConfig` — output ceiling, fallback cache multipliers.
        client: The underlying OpenAI SDK client, pointed at OpenRouter.
    """

    def __init__(self, catalog: ModelCatalog, call_cfg, client=None):
        """Build a client.

        Args:
            catalog: The model catalog to route and price against.
            call_cfg: A `CallConfig`.
            client: An SDK client to use instead of constructing one. Mainly
                for tests.

        Raises:
            RuntimeError: No `client` was given and OPENROUTER_API_KEY is unset.
                Every call here is a real, billed API call, so this fails early
                rather than at the first request.
        """
        if client is None and not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("Set OPENROUTER_API_KEY — every call here is a real, "
                               "billed API call.")
        self.catalog = catalog
        self.cfg = call_cfg
        # Generous retries: a transient upstream overload on one call would
        # otherwise score that example 0 and understate a candidate. Responses
        # are STREAMED, so the timeout that matters is the inter-chunk gap:
        # OpenRouter sends keepalive comments while a model thinks, which means
        # a healthy connection is never silent for long — and a socket that
        # died during a laptop sleep is detected in minutes, not the hour a
        # whole-call deadline allowed (one slept eval sat wedged for exactly
        # that reason).
        self.client = client or openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=8,
            timeout=httpx.Timeout(180.0, connect=30.0))

    def call(self, model: str, prompt, system: Optional[str] = None,
             tools: Optional[list[str]] = None, effort: Optional[str] = None,
             schema=None) -> ApiResponse:
        """Make one model call.

        The prompt and system prompt carry a cache breakpoint, which providers
        that need explicit breakpoints (Anthropic) honor and the rest ignore in
        favor of their automatic caching. Either way a repeated prefix to the
        SAME model bills at a deep discount and shows up in `usage` as
        "cache_read"; a different model is always a fresh miss.

        Args:
            model: OpenRouter model id to call.
            prompt: The user message. Stringified, so any object is accepted.
            system: Optional system prompt.
            tools: Server-side tool names to enable — "web_search", "web_fetch"
                and/or "subagent". See TOOL_DEFS.
            effort: Thinking depth — "low" through "max". Ignored on models that
                don't support reasoning.
            schema: Constrains the final text to JSON. Either a Pydantic model
                class or a raw JSON Schema dict.

        Returns:
            An ApiResponse carrying the final text, token usage, and — when
            OpenRouter reported it — the actually-billed cost. `truncated` is
            set when the output ceiling cut the reply off.
        """
        request = self._request(model, prompt, system, tools, effort, schema)
        try:
            return self._stream(request)
        except openai.BadRequestError as error:
            # Some endpoints cannot have reasoning disabled (Gemini flash-lite,
            # pro serving aliases) and 400 on our default {"effort": "none"}.
            # Drop the reasoning field and let the endpoint run its mandatory
            # default — the billed cost reports what it actually spent.
            if "reasoning" in str(error).lower() and request["extra_body"].pop("reasoning", None):
                return self._stream(request)
            raise

    def _stream(self, request) -> ApiResponse:
        """Issue one request and assemble its streamed reply.

        The SDK only retries failures BEFORE the stream opens; a connection
        that drops mid-stream raises out of the chunk iterator instead. That
        is retried here — a fresh generation, honestly re-billed — because a
        broken stream says nothing about the candidate being evaluated.

        Args:
            request: `_request` output.

        Returns:
            The assembled ApiResponse.
        """
        failure = None
        for _ in range(3):
            try:
                return self._consume(self.client.chat.completions.create(**request))
            except openai.APIStatusError:
                raise                              # a real rejection, not a drop
            except (openai.APIError, httpx.HTTPError) as error:
                failure = error
        raise failure

    def _consume(self, stream) -> ApiResponse:
        """Fold a chunk stream into one ApiResponse.

        Args:
            stream: The SDK's chunk iterator.

        Returns:
            The assembled response. OpenRouter delivers the usage (and billed
            cost) on the final chunk when `usage: {include: true}` is set.
        """
        parts, annotations, finish, usage = [], [], None, None
        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None and getattr(delta, "content", None):
                    parts.append(delta.content)
                annotations.extend(getattr(delta, "annotations", None) or [])
                finish = choice.finish_reason or finish
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
        counts, cost = _usage_of(usage)
        return ApiResponse(text="".join(parts), blocks=annotations,
                           usage=counts, truncated=finish == "length", cost=cost)

    def parse(self, model: str, prompt: str, schema_model: type[BaseModel]) -> BaseModel:
        """Make one structured call and validate the reply into a Pydantic model.

        The reply is constrained to the schema as long as it fits in
        `max_output_tokens` — a reply cut off at the ceiling is invalid JSON, so
        keep expected output well under it.

        Args:
            model: OpenRouter model id to call.
            prompt: The user message.
            schema_model: The Pydantic class constraining and typing the reply.

        Returns:
            An instance of `schema_model`.

        Raises:
            pydantic.ValidationError: The reply did not parse — in practice a
                refusal, or output truncated at the ceiling.
        """
        return schema_model.model_validate_json(self.call(model, prompt, schema=schema_model).text)

    def _request(self, model, prompt, system, tools, effort, schema) -> dict:
        """Assemble the chat completions request body. See `call` for the arguments."""
        # cache_control marks a cache breakpoint for providers that want one
        # (Anthropic); the rest cache automatically and ignore it.
        breakpoint_text = lambda text: [{"type": "text", "text": str(text),
                                         "cache_control": {"type": "ephemeral"}}]
        messages = []
        if system:
            messages.append({"role": "system", "content": breakpoint_text(system)})
        messages.append({"role": "user", "content": breakpoint_text(prompt)})

        request = {
            "model": model,
            "messages": messages,
            "max_tokens": self.cfg.max_output_tokens,
            "stream": True,
            # extra_body carries OpenRouter's extensions to the OpenAI schema.
            "extra_body": {"usage": {"include": True}},   # report the billed cost
        }
        if tools:
            defs = [TOOL_DEFS[TOOL_ALIASES.get(name, name)] for name in tools]
            request["tools"] = [d for i, d in enumerate(defs) if d not in defs[:i]]
        if self.cfg.plugins:
            request["extra_body"]["plugins"] = [PLUGIN_DEFS[p] for p in self.cfg.plugins]
        if self.catalog.thinks(model):
            # Default is NO thinking — the strategy is the only knob — rather
            # than the provider defaults (usually "medium"), which would bill
            # invisible reasoning tokens on every plain call.
            request["extra_body"]["reasoning"] = (
                {"effort": effort} if effort else {"effort": "none"})

        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema = schema.model_json_schema()
            # strict=False: workflow-written schemas rarely meet strict mode's
            # extra rules (additionalProperties etc.), and a 400 would score the
            # example 0 — best-effort adherence is the right trade.
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": False, "schema": schema}}
        return request


def _usage_of(usage) -> tuple[dict, Optional[float]]:
    """Extract token counts and OpenRouter's billed cost from a usage object.

    OpenRouter counts cached tokens INSIDE prompt_tokens, so they are carved out
    here — the keys below are disjoint and can be priced independently. Cache
    writes are not itemized separately (the billed `cost` accounts for them),
    so "cache_write" is always 0.

    Args:
        usage: The usage object from the final stream chunk, or None if the
            stream ended without one (counts then read zero and cost None, so
            the caller's catalog fallback prices what it can).

    Returns:
        (counts keyed "input"/"output"/"cache_write"/"cache_read", billed USD
        or None).
    """
    if usage is None:
        return {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}, None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    counts = {
        "input": usage.prompt_tokens - cached,
        "output": usage.completion_tokens,
        "cache_write": 0,
        "cache_read": cached,
    }
    return counts, getattr(usage, "cost", None)
