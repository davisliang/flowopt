"""The wiring every entry point needs, built once.

A run is always the same three things: the config, the model catalog derived from
it, and the client that calls those models. Spelling that out at each entry point
(CLI, notebook, experiment script, the agent's dev evaluator) meant four copies of
the same two lines, and four chances for them to drift.
"""
from dataclasses import dataclass

from .client import ModelClient
from .config import load_config
from .grading import Grader
from .models import ModelCatalog
from .runtime import Evaluator


@dataclass
class Session:
    """One run's configuration, and the client it talks to models through.

    Attributes:
        cfg: The loaded config for this run.
        client: A ModelClient built from it, holding the model catalog.
    """
    cfg: object
    client: ModelClient

    @classmethod
    def from_config(cls, cfg) -> "Session":
        """Build a session around an already-loaded config.

        Args:
            cfg: A `Config`, or the OmegaConf equivalent.

        Returns:
            The session.

        Raises:
            RuntimeError: ANTHROPIC_API_KEY is unset.
        """
        return cls(cfg=cfg, client=ModelClient(ModelCatalog.from_config(cfg), cfg.call))

    @classmethod
    def load(cls, task: str = "clinical_notes", overrides: list[str] = ()) -> "Session":
        """Load a task's config and build a session — the usual entry point.

        Args:
            task: Name of a file under `config/task/`, without the extension.
            overrides: OmegaConf dotlist entries, e.g. `["designer.rounds=1"]`.

        Returns:
            The session.
        """
        return cls.from_config(load_config(task, overrides))

    @property
    def catalog(self) -> ModelCatalog:
        """The models available to this run, and their prices."""
        return self.client.catalog

    def evaluator(self, grader: Grader) -> Evaluator:
        """Build an evaluator that scores candidates with the given grader.

        Args:
            grader: How to score a returned answer — usually `benchmark.grader`.

        Returns:
            An Evaluator wired to this session's client and runtime limits.
        """
        return Evaluator(self.client, grader, self.cfg.runtime)
