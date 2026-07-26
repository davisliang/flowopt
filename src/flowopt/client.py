"""The one place anything in this project reaches a model.

Every model call — the task analyzer, the dataset generator, the judge, and every
call a candidate workflow makes — goes through `ModelClient.call`. That single
chokepoint is what makes cost measurable no matter what a workflow's code does.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import anthropic
from pydantic import BaseModel

from .models import ModelCatalog

# Server-side tools. They run on Anthropic's side, so there is nothing to execute
# locally — the model uses them and the results come back in the same reply.
TOOL_DEFS = {
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
    # allowed_callers=["direct"] is REQUIRED for the cheap model: the _20260209
    # web tools default to being called from inside code execution (that's how
    # dynamic filtering works), and haiku can't do programmatic tool calling —
    # without this every web_search call on the default model dies with a 400.
    "web_search": {"type": "web_search_20260209", "name": "web_search",
                   "allowed_callers": ["direct"]},
    # Fetches a URL already present in the conversation (it won't discover new
    # ones — pair it with web_search for that). Same direct-caller requirement.
    "web_fetch": {"type": "web_fetch_20260209", "name": "web_fetch",
                  "allowed_callers": ["direct"]},
}

# The _20260209 web tools run code execution under the hood for dynamic
# filtering, so declaring code_execution alongside either one gives the model a
# second execution environment and confuses it. A workflow may use the web tools
# OR code_execution, not both in one call — enforced in CallMeter.


@dataclass
class ApiResponse:
    """The result of one completed call to the model API.

    Attributes:
        text: The final answer text — the last API response's text blocks joined.
        blocks: Every content block from every API response the call made (tool
            uses, tool results, text), in order. Kept whole for the trace.
        usage: Token counts with keys "input", "output", "cache_write" and
            "cache_read".
        truncated: True when the call ran out of tool turns while the model was
            still working — `text` is then a partial turn, not a finished answer.
    """
    text: str
    blocks: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    truncated: bool = False


class ModelClient:
    """A thin, cache-aware wrapper over the Anthropic Messages API.

    Holds the model catalog and per-call settings so callers pass neither around.

    Attributes:
        catalog: The models this client may call, and their prices.
        cfg: A `CallConfig` — output ceiling, tool-turn cap, cache multipliers.
        client: The underlying Anthropic SDK client.
    """

    def __init__(self, catalog: ModelCatalog, call_cfg, client=None):
        """Build a client.

        Args:
            catalog: The model catalog to route and price against.
            call_cfg: A `CallConfig`.
            client: An Anthropic SDK client to use instead of constructing one.
                Mainly for tests.

        Raises:
            RuntimeError: No `client` was given and ANTHROPIC_API_KEY is unset.
                Every call here is a real, billed API call, so this fails early
                rather than at the first request.
        """
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY — every call here is a real API call.")
        self.catalog = catalog
        self.cfg = call_cfg
        # A transient 529 "Overloaded" on one call would otherwise score that
        # example 0 and understate a candidate. Retry generously (the SDK backs off
        # exponentially) so infra load doesn't leak into the accuracy signal.
        self.client = client or anthropic.Anthropic(max_retries=8)

    def call(self, model: str, prompt, system: Optional[str] = None,
             tools: Optional[list[str]] = None, effort: Optional[str] = None,
             schema=None) -> ApiResponse:
        """Make one model call, resuming it while a server-side tool pauses the turn.

        The prompt and system prompt carry a cache breakpoint, so resending the
        SAME prompt to the SAME model bills the repeat at ~10% of the input rate.
        The cache is keyed by model, so a different model is always a fresh miss,
        and prompts under the model's floor (~1-4k tokens) don't cache at all.

        Args:
            model: API model id to call.
            prompt: The user message. Stringified, so any object is accepted.
            system: Optional system prompt.
            tools: Server-side tool names to enable — "code_execution" and/or
                "web_search". See TOOL_DEFS.
            effort: Thinking depth — "low" through "max". Ignored on models that
                don't support thinking.
            schema: Constrains the final text to JSON. Either a Pydantic model
                class or a raw JSON Schema dict. Tool calls in the same reply are
                unaffected.

        Returns:
            An ApiResponse carrying the final text, every content block, and the
            summed token usage across all turns of the call. `truncated` is set
            when the tool-turn cap cut the call off mid-work.
        """
        request = self._request(model, prompt, system, tools, effort, schema)
        turn_texts, blocks = [], []
        usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
        truncated = False

        for _ in range(self.cfg.max_tool_turns):
            # Streamed because max_output_tokens is large: the SDK refuses a
            # non-streaming request whose ceiling could outlive the HTTP timeout.
            # get_final_message() reassembles the Message a create() would return.
            with self.client.messages.stream(**request) as stream:
                message = stream.get_final_message()

            for key, value in _usage_of(message).items():
                usage[key] += value
            blocks.extend(message.content)
            # Join the text blocks WITHIN this response (citations split one
            # message into several), but keep responses apart — see below.
            turn_texts.append("".join(b.text for b in message.content if b.type == "text"))

            if message.stop_reason != "pause_turn":
                break
            request["messages"].append({"role": "assistant", "content": message.content})
        else:
            # Every turn ended in pause_turn: the cap cut the call off with the
            # model still working. The text below is a partial turn — flag it
            # rather than pass it off as a finished answer.
            truncated = True

        # The LAST response holds the answer; earlier ones are the model working
        # up to it. Concatenating across responses would splice that preamble onto
        # the answer — and with `schema` set, produce unparseable JSON.
        return ApiResponse(text=turn_texts[-1] if turn_texts else "", blocks=blocks,
                           usage=usage, truncated=truncated)

    def parse(self, model: str, prompt: str, schema_model: type[BaseModel]) -> BaseModel:
        """Make one structured call and validate the reply into a Pydantic model.

        The reply is guaranteed to match the schema as long as it fits in
        `max_output_tokens` — a reply cut off at the ceiling is invalid JSON, so
        keep expected output well under it.

        Args:
            model: API model id to call.
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
        """Assemble the Messages API request body. See `call` for the arguments."""
        # cache_control "ephemeral" = an auto-expiring cache entry (typically 5m).
        request = {
            "model": model,
            "max_tokens": self.cfg.max_output_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": str(prompt),
                 "cache_control": {"type": "ephemeral"}}]}],
        }
        if system:
            request["system"] = [{"type": "text", "text": system,
                                  "cache_control": {"type": "ephemeral"}}]
        if tools:
            request["tools"] = [TOOL_DEFS[name] for name in tools]

        if effort and self.catalog.thinks(model):
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": effort}
        else:
            request["thinking"] = {"type": "disabled"}   # default: the strategy is the only knob

        if schema is not None:
            # Constrains ONLY the text the model writes at the end — tool calls
            # and tool results in the same reply are untouched. Takes a Pydantic
            # class (what this package uses internally) or a raw JSON Schema dict
            # (what a workflow program passes, since the sandbox has no pydantic).
            request["output_format"] = (
                schema if isinstance(schema, type) and issubclass(schema, BaseModel)
                else {"type": "json_schema", "schema": schema})
        return request


def _usage_of(message) -> dict:
    """Extract one response's token counts, split so cached and fresh tokens can
    be priced differently.

    Args:
        message: An Anthropic SDK Message.

    Returns:
        Counts keyed "input", "output", "cache_write", "cache_read".
    """
    return {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
        "cache_write": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }
