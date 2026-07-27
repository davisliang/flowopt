"""The model catalog: which models a run may use, what they cost, and what they
can do.

`config.models` is just OpenRouter model ids. Everything else about them —
prices (OpenRouter), capability and speed measurements (Artificial Analysis) —
is fetched, so the facts the search routes on are the providers' own, not a
hand-maintained table that can drift.
"""
from dataclasses import dataclass
from typing import Optional

from . import catalog as feeds


@dataclass
class ModelSpec:
    """One model the optimizer may use, with the facts needed to price and route it.

    Attributes:
        id: The OpenRouter model id, e.g. "anthropic/claude-haiku-4.5".
        price_in: USD per 1,000,000 input tokens.
        price_out: USD per 1,000,000 output tokens.
        thinks: Whether the model supports the reasoning-effort parameter. A
            workflow asking for `effort` on a model that can't think is ignored
            rather than erroring.
        price_cache_read: USD per 1M cached input tokens read, when OpenRouter
            prices it; None falls back to the config multiplier.
        price_cache_write: Likewise for writing the prompt cache.
        context_length: The model's context window, when known.
        aa: Artificial Analysis's measurements — intelligence / coding / math
            indices and tokens_per_second — or None when AA doesn't cover it.
    """
    id: str
    price_in: float
    price_out: float
    thinks: bool = False
    price_cache_read: Optional[float] = None
    price_cache_write: Optional[float] = None
    context_length: Optional[int] = None
    aa: Optional[dict] = None


@dataclass
class ModelCatalog:
    """The models available to a run, and the pricing used to bill them.

    Attributes:
        specs: ModelSpec records, ordered cheapest to most expensive.
        cache_write_multiplier: Fallback multiple of the input rate billed for
            cache writes, for models OpenRouter has no cache price for.
        cache_read_multiplier: Likewise for cache reads.
    """
    specs: list[ModelSpec]
    cache_write_multiplier: float
    cache_read_multiplier: float

    @classmethod
    def from_config(cls, cfg, openrouter: Optional[dict] = None,
                    aa: Optional[dict] = None) -> "ModelCatalog":
        """Build the catalog for a config's model list.

        Args:
            cfg: A `Config` (or OmegaConf equivalent) with `models` (OpenRouter
                ids) and `call`.
            openrouter: `catalog.openrouter_models()` output, for tests. None
                fetches (through the disk cache).
            aa: `catalog.aa_models()` output, for tests. None fetches.

        Returns:
            The catalog, specs sorted cheapest first.

        Raises:
            ValueError: A configured id is not on OpenRouter — a typo caught at
                load time, not as a 404 mid-run.
        """
        openrouter = openrouter if openrouter is not None else feeds.openrouter_models()
        aa = aa if aa is not None else feeds.aa_models()

        specs = []
        for model_id in cfg.models:
            record = openrouter.get(model_id)
            if record is None:
                raise ValueError(f"model '{model_id}' is not on OpenRouter — "
                                 f"check the id against https://openrouter.ai/models")
            pricing = record.get("pricing", {})
            per_m = lambda key, p=pricing: (float(p[key]) * 1_000_000
                                            if p.get(key) not in (None, "") else None)
            specs.append(ModelSpec(
                id=model_id,
                price_in=per_m("prompt") or 0.0,
                price_out=per_m("completion") or 0.0,
                thinks="reasoning" in (record.get("supported_parameters") or []),
                price_cache_read=per_m("input_cache_read"),
                price_cache_write=per_m("input_cache_write"),
                context_length=record.get("context_length"),
                aa=feeds.aa_summary(model_id, aa),
            ))
        # Cheapest first, whatever order they were picked in: `default` and the
        # "cheap -> expensive" menus everywhere rely on it.
        specs.sort(key=lambda s: (s.price_in, s.price_out))
        return cls(specs=specs,
                   cache_write_multiplier=cfg.call.cache_write_multiplier,
                   cache_read_multiplier=cfg.call.cache_read_multiplier)

    @property
    def ids(self) -> list[str]:
        """Model ids, cheapest to most expensive.

        This is both the search pool and the menu a workflow routes over — it is
        handed to candidate programs as `MODELS`.
        """
        return [m.id for m in self.specs]

    @property
    def default(self) -> str:
        """The model a workflow starts on: the cheapest. It may escalate itself."""
        return self.ids[0]

    def spec(self, model_id: str) -> Optional[ModelSpec]:
        """Look up one model's record.

        Args:
            model_id: An OpenRouter model id.

        Returns:
            Its ModelSpec, or None if this catalog has no such model.
        """
        for spec in self.specs:
            if spec.id == model_id:
                return spec
        return None

    def thinks(self, model_id: str) -> bool:
        """Report whether a model supports the effort / reasoning parameters.

        Args:
            model_id: An OpenRouter model id.

        Returns:
            True if the model can think; False for unknown models too, so an
            `effort` request on one is simply dropped.
        """
        spec = self.spec(model_id)
        return bool(spec and spec.thinks)

    def resolve(self, model_id: Optional[str]) -> str:
        """Map a requested model name onto one this catalog actually has.

        Model-written code routes by name and may invent one, so an unknown or
        missing name falls back to the default rather than failing the query.

        Args:
            model_id: The requested model id, possibly unknown or None.

        Returns:
            `model_id` if the catalog has it, otherwise `self.default`.
        """
        return model_id if self.spec(model_id) else self.default

    def cost_usd(self, model_id: str, usage: dict) -> float:
        """Price one call's token usage in US dollars.

        Cache-aware, using OpenRouter's per-model cache prices when it lists
        them and the config multipliers otherwise. This is the estimate used
        when OpenRouter's own billed cost isn't attached to a response —
        metered calls prefer the billed number.

        Args:
            model_id: The model that served the call. Must be in this catalog.
            usage: Token counts with keys "input", "output", "cache_write" and
                "cache_read".

        Returns:
            The cost of the call in USD.
        """
        spec = self.spec(model_id)
        write = (spec.price_cache_write if spec.price_cache_write is not None
                 else spec.price_in * self.cache_write_multiplier)
        read = (spec.price_cache_read if spec.price_cache_read is not None
                else spec.price_in * self.cache_read_multiplier)
        tokens = (usage["input"] * spec.price_in
                  + usage["cache_write"] * write
                  + usage["cache_read"] * read
                  + usage["output"] * spec.price_out)
        return tokens / 1_000_000

    def describe(self, model_id: str) -> str:
        """One line saying what a model costs and what it is measured to do.

        Used everywhere a human or an agent is shown the menu — the run prompt,
        the research prompt, the logs.

        Args:
            model_id: An OpenRouter model id in this catalog.

        Returns:
            E.g. "anthropic/claude-haiku-4.5: $1/M in, $5/M out — intelligence
            31, coding 28, math 55, ~180 tok/s (Artificial Analysis)".
        """
        spec = self.spec(model_id)
        line = f"{spec.id}: ${spec.price_in:g}/M in, ${spec.price_out:g}/M out"
        if spec.aa is None and spec.id.endswith("-pro"):
            # OpenRouter's pro serving aliases: same weights and per-token price
            # as the base model, but reasoning always on and much heavier — so a
            # call costs several times more in practice. AA doesn't measure the
            # alias, and borrowing the base numbers would misstate it.
            line += (" — pro serving mode of the base model: same rates, mandatory "
                     "heavy reasoning, so far higher effective cost per call")
        if spec.aa:
            parts = [f"{key} {round(value)}" for key, value in
                     (("intelligence", spec.aa.get("intelligence")),
                      ("coding", spec.aa.get("coding")),
                      ("math", spec.aa.get("math"))) if value is not None]
            speed = spec.aa.get("tokens_per_second")
            if speed:
                parts.append(f"~{round(speed)} tok/s")
            if parts:
                line += " — " + ", ".join(parts) + " (Artificial Analysis)"
        return line
