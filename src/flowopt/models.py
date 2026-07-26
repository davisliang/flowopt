"""The model catalog: which models exist, what they cost, and what they can do."""
from dataclasses import dataclass
from typing import Optional

from .config import ModelSpec


@dataclass
class ModelCatalog:
    """The models available to a run, and the pricing used to bill them.

    Built from `config.models`, so cost and capability are looked up in the same
    list the search routes over — a price and the model it prices cannot drift
    apart.

    Attributes:
        specs: ModelSpec records, ordered cheapest to most expensive.
        cache_write_multiplier: Multiple of the input rate billed for cache writes.
        cache_read_multiplier: Multiple of the input rate billed for cache reads.
    """
    specs: list[ModelSpec]
    cache_write_multiplier: float
    cache_read_multiplier: float

    @classmethod
    def from_config(cls, cfg) -> "ModelCatalog":
        """Build the catalog from a loaded config.

        Args:
            cfg: A `Config` (or OmegaConf equivalent) with `models` and `call`.

        Returns:
            The catalog described by `cfg.models`.
        """
        return cls(specs=list(cfg.models),
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
            model_id: An API model id.

        Returns:
            Its ModelSpec, or None if this catalog has no such model.
        """
        for spec in self.specs:
            if spec.id == model_id:
                return spec
        return None

    def thinks(self, model_id: str) -> bool:
        """Report whether a model supports the effort / thinking parameters.

        Args:
            model_id: An API model id.

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

        Cache-aware: writing the prompt cache costs a little more than fresh
        input, and reading it costs ~90% less.

        Args:
            model_id: The model that served the call. Must be in this catalog.
            usage: Token counts with keys "input", "output", "cache_write" and
                "cache_read".

        Returns:
            The cost of the call in USD.
        """
        spec = self.spec(model_id)
        tokens = (usage["input"] * spec.price_in
                  + usage["cache_write"] * spec.price_in * self.cache_write_multiplier
                  + usage["cache_read"] * spec.price_in * self.cache_read_multiplier
                  + usage["output"] * spec.price_out)
        return tokens / 1_000_000
