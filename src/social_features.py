"""Compute transparent social relevance features for financial Q&A answers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from . import config
from .preprocessing import tokenize
from .sentiment import sentiment_alignment


def minmax(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to [0, 1], returning zeros for constants."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - lo) / (hi - lo)


def topical_from_retrieval(scores: pd.Series) -> pd.Series:
    """Normalize retrieval scores into topical_score."""
    return minmax(scores)


def tag_match_score(query: str, tags: str) -> float:
    """Explainable Jaccard overlap between query terms and post tags."""
    query_terms = set(tokenize(query))
    tag_terms = set(tokenize(tags))
    if not query_terms or not tag_terms:
        return 0.0
    return len(query_terms & tag_terms) / len(query_terms | tag_terms)


def add_social_features(results: pd.DataFrame, query: str, half_life_days: int | None = None, graph_ctx=None) -> pd.DataFrame:
    """Add required social-search features to candidate results.

    When ``graph_ctx`` (a SocialGraphContext) is supplied, the social-GRAPH
    influence feature ``author_influence_score`` is computed from PageRank over
    the user interaction graph; otherwise it defaults to zero.
    """
    half_life_days = half_life_days or config.FRESHNESS_HALF_LIFE_DAYS
    out = results.copy()
    out["topical_score"] = topical_from_retrieval(out.get("retrieval_score", 0))
    accepted_bonus = out.get("accepted_answer", False).astype(bool).astype(float) * 1.0
    community_raw = np.log1p(pd.to_numeric(out.get("answer_score", 0), errors="coerce").clip(lower=0).fillna(0)) + accepted_bonus
    out["community_score"] = minmax(community_raw)
    engagement_raw = (
        np.log1p(pd.to_numeric(out.get("comment_count", 0), errors="coerce").fillna(0))
        + 0.5 * np.log1p(pd.to_numeric(out.get("question_view_count", 0), errors="coerce").fillna(0))
        + np.log1p(pd.to_numeric(out.get("answer_count", 0), errors="coerce").fillna(0))
    )
    out["engagement_score"] = minmax(engagement_raw)
    out["credibility_score"] = minmax(np.log1p(pd.to_numeric(out.get("author_reputation", 0), errors="coerce").fillna(0)))
    if graph_ctx is not None:
        from .social_graph import author_influence_series

        out["author_influence_score"] = author_influence_series(out, graph_ctx).values
    else:
        out["author_influence_score"] = 0.0
    age_days = pd.to_numeric(out.get("age_days", 3650), errors="coerce").fillna(3650).clip(lower=0)
    out["freshness_score"] = np.exp(-age_days / float(half_life_days))
    out["tag_match_score"] = out.get("tags", "").fillna("").map(lambda tags: tag_match_score(query, tags))
    out["sentiment_alignment_score"] = out.get("combined_text", "").fillna("").map(
        lambda text: sentiment_alignment(query, text)
    )
    answer_lengths = out.get("answer_body", "").fillna("").str.len().clip(lower=1)
    out["clarity_score"] = 1 - minmax(np.log1p(answer_lengths))
    investing_terms = {"etf", "stock", "stocks", "portfolio", "bond", "investing", "retirement", "tax"}
    risk_terms = {"scam", "fraud", "debt", "risk", "avoid", "warning", "credit-card"}
    out["technical_score"] = out.get("combined_text", "").fillna("").map(
        lambda text: min(1.0, len(set(tokenize(text)) & investing_terms) / 4)
    )
    out["risk_warning_score"] = out.get("combined_text", "").fillna("").map(
        lambda text: min(1.0, len(set(tokenize(text)) & risk_terms) / 4)
    )
    return out

