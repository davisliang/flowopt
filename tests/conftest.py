"""Shared test setup: the model feeds are served from checked-in fixtures.

Every test runs offline. `flowopt.catalog`'s two fetchers are replaced for the
whole session with snapshots of the real OpenRouter and Artificial Analysis
responses (tests/fixtures/), so catalog construction — and everything downstream
of it — behaves exactly as in production, without a network or keys.
"""
import json
import pathlib

import pytest

from flowopt import catalog as feeds

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def offline_model_feeds(monkeypatch):
    openrouter = json.loads((FIXTURES / "openrouter_models.json").read_text())
    aa = json.loads((FIXTURES / "aa_models.json").read_text())
    monkeypatch.setattr(feeds, "openrouter_models", lambda refresh=False: openrouter)
    monkeypatch.setattr(feeds, "aa_models", lambda refresh=False: aa)
