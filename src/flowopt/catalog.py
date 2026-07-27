"""Fetching what is known about the models: OpenRouter's catalog, and
Artificial Analysis's measurements of them.

Everything this project calls goes through OpenRouter, so OpenRouter's public
model list is the source of truth for what exists and what it costs — prices
are pulled from there, never hand-maintained. Artificial Analysis supplies the
part OpenRouter doesn't know: how good each model actually is (intelligence /
coding / math indices) and how fast it runs. The two are joined by model slug
and the result annotates every `ModelSpec`, which is how the design agent knows
what it is routing between.

Both feeds are cached on disk so a run doesn't refetch them, and a stale cache
is better than no catalog — if the network is down, whatever was fetched last
time is used.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

CACHE_TTL_SECONDS = 24 * 3600


def _cache_dir() -> Path:
    """Where the fetched catalogs are cached. Override with FLOWOPT_CACHE_DIR."""
    return Path(os.environ.get("FLOWOPT_CACHE_DIR",
                               Path.home() / ".cache" / "flowopt"))


def _fetch_json(url: str, headers: Optional[dict] = None) -> dict:
    """GET a JSON document.

    Args:
        url: What to fetch.
        headers: Extra request headers (the AA key travels here).

    Returns:
        The parsed body.
    """
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _cached_fetch(name: str, fetch, refresh: bool = False) -> dict:
    """Serve `fetch()` through the on-disk cache.

    Fresh cache wins; otherwise fetch and rewrite it; if the fetch fails but a
    stale cache exists, the stale copy is served rather than failing the run.

    Args:
        name: Cache file name under the cache directory.
        fetch: Zero-argument callable that does the real fetch.
        refresh: True skips the freshness check (still falls back on failure).

    Returns:
        The document, from wherever it could be had.

    Raises:
        RuntimeError: The fetch failed and no cached copy exists at all.
    """
    path = _cache_dir() / name
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS
    if fresh and not refresh:
        return json.loads(path.read_text())
    try:
        data = fetch()
    except (urllib.error.URLError, OSError, ValueError) as error:
        if path.exists():                      # stale beats nothing
            return json.loads(path.read_text())
        raise RuntimeError(f"could not fetch {name} and no cached copy exists: "
                           f"{error}") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def openrouter_models(refresh: bool = False) -> dict:
    """OpenRouter's model catalog, keyed by model id.

    Public — no key needed. Each value is OpenRouter's raw record: `pricing`
    (USD per TOKEN, as strings), `context_length`, `supported_parameters`, ...

    Args:
        refresh: True refetches even if the cache is fresh.

    Returns:
        `{model_id: record}` for every model OpenRouter serves.
    """
    raw = _cached_fetch("openrouter_models.json",
                        lambda: _fetch_json(OPENROUTER_MODELS_URL), refresh)
    return {m["id"]: m for m in raw["data"]}


def aa_models(refresh: bool = False) -> dict:
    """Artificial Analysis's model measurements, keyed by AA slug.

    Needs ARTIFICIAL_ANALYSIS_API_KEY. Without it (or if the fetch fails with
    no cache) this returns {} rather than blocking the run — capabilities are
    an annotation, not a requirement.

    Args:
        refresh: True refetches even if the cache is fresh.

    Returns:
        `{slug: record}`, or {} when unavailable.
    """
    key = os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY")
    if not key:
        return {}
    try:
        raw = _cached_fetch("aa_models.json",
                            lambda: _fetch_json(AA_MODELS_URL, {"x-api-key": key}),
                            refresh)
    except RuntimeError:
        return {}
    data = raw.get("data", raw)
    return {m["slug"]: m for m in data if m.get("slug")}


def _norm(slug: str) -> str:
    """Normalize a model slug for matching: lowercase, separators unified."""
    return re.sub(r"[._/]", "-", str(slug).lower())


# AA lists reasoning models once per effort configuration. When the base slug
# itself is absent, these suffixes are tried cheapest-thinking-first, so the
# match describes the model, not its most expensive mode.
_AA_VARIANT_SUFFIXES = ("", "-standard", "-medium", "-low", "-high", "-max")


def aa_summary(model_id: str, aa: dict) -> Optional[dict]:
    """Artificial Analysis's measurements for one OpenRouter model, condensed.

    Joins on slug: "anthropic/claude-sonnet-5" matches AA's "claude-sonnet-5",
    or an effort variant of it when only those exist.

    Args:
        model_id: The OpenRouter model id.
        aa: `aa_models()` output.

    Returns:
        `{name, intelligence, coding, math, tokens_per_second}` (fields None
        when AA has no number), or None when AA doesn't cover the model.
    """
    if not aa:
        return None
    base = _norm(model_id.split("/", 1)[-1])
    by_norm = {_norm(slug): m for slug, m in aa.items()}
    record = next((by_norm[base + s] for s in _AA_VARIANT_SUFFIXES
                   if base + s in by_norm), None)
    if record is None:
        # The two sites order name parts differently ("claude-haiku-4-5" vs
        # AA's "claude-4-5-haiku"), so fall back to same-tokens-any-order.
        tokens = sorted(base.split("-"))
        record = next((m for slug, m in by_norm.items()
                       if sorted(slug.split("-")) == tokens), None)
    if record is None:
        return None
    evals = record.get("evaluations") or {}
    return {
        "name": record.get("name"),
        "intelligence": evals.get("artificial_analysis_intelligence_index"),
        "coding": evals.get("artificial_analysis_coding_index"),
        "math": evals.get("artificial_analysis_math_index"),
        "tokens_per_second": record.get("median_output_tokens_per_second"),
    }
