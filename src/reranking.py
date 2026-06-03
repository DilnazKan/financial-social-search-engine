"""Social reranking: transparent weighted fusion of topical + social signals.

The re-ranker first retrieves topical candidates with a classical model (BM25 /
TF-IDF) and then reorders ONLY those candidates using social-search signals:
community approval, engagement, author credibility, social-GRAPH influence,
freshness, tag match and sentiment alignment. When a SEARCHER identity is
supplied it additionally personalises the order using the social graph
(proximity to the searcher's network and affinity with their communities).
"""

from __future__ import annotations

import pandas as pd

from . import config
from .social_features import add_social_features


def get_weights(profile: str = "default", searcher: bool = False) -> dict[str, float]:
    """Return ranking weights for a profile, adding searcher-graph weights."""
    weights = dict(config.PROFILE_WEIGHTS.get(profile, config.DEFAULT_SOCIAL_WEIGHTS))
    if searcher:
        weights.update(config.SEARCHER_WEIGHTS)
    return weights


def apply_social_reranking(
    candidates: pd.DataFrame,
    query: str,
    profile: str = "default",
    ablate: str | None = None,
    graph_ctx=None,
    searcher=None,
) -> pd.DataFrame:
    """Compute social features and the final weighted score for candidates.

    Parameters
    ----------
    graph_ctx : SocialGraphContext, optional
        Enables the social-graph influence feature.
    searcher : ResolvedSearcher, optional
        Enables social-graph personalisation (proximity + community affinity)
        and selects the searcher's intent profile.
    """
    rows = add_social_features(candidates, query, graph_ctx=graph_ctx)

    use_profile = profile
    is_searcher = False
    if searcher is not None and graph_ctx is not None:
        from .social_graph import add_searcher_features

        rows = add_searcher_features(rows, graph_ctx, searcher)
        use_profile = searcher.intent_profile
        is_searcher = True

    weights = get_weights(use_profile, searcher=is_searcher).copy()
    if ablate and ablate in weights:
        weights[ablate] = 0.0
    weight_sum = sum(weights.values()) or 1.0

    rows["final_score"] = 0.0
    for feature, weight in weights.items():
        if feature not in rows:
            rows[feature] = 0.0
        rows["final_score"] += (weight / weight_sum) * rows[feature].fillna(0)

    rows = rows.sort_values(["final_score", "topical_score"], ascending=False).reset_index(drop=True)
    rows["rank"] = range(1, len(rows) + 1)
    label = searcher.name if is_searcher else profile
    rows["run"] = f"social_{label}" + (f"_without_{ablate}" if ablate else "")
    return rows
