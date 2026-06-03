"""Central configuration for the Financial Social Search Engine.

This module centralises every path, weight and tunable constant so that the
ranking model, the social-graph layer and the evaluation are fully transparent
and reproducible. Nothing in the pipeline hardcodes a path outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QRELS_DIR = DATA_DIR / "qrels"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_TABLES_DIR = PROJECT_ROOT / "report_assets" / "tables"
REPORT_FIGURES_DIR = PROJECT_ROOT / "report_assets" / "figures"

PROCESSED_CSV = PROCESSED_DIR / "social_finance_docs.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "social_finance_docs.parquet"
SAMPLE_CSV = SAMPLE_DIR / "social_finance_sample.csv"
SOCIAL_QUERIES_CSV = QRELS_DIR / "social_eval_queries.csv"
JUDGMENT_TEMPLATE_CSV = QRELS_DIR / "social_judgments_template.csv"
# Model-independent reference judgments (replaces the old circular example file).
REFERENCE_JUDGMENTS_CSV = QRELS_DIR / "social_reference_judgments.csv"

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATES = 100
FRESHNESS_HALF_LIFE_DAYS = 730

# --------------------------------------------------------------------------- #
# Ranking weights
# --------------------------------------------------------------------------- #
# The social re-ranker is a transparent linear combination. Topical relevance is
# always the largest weight (an answer must first be about the query); the
# remaining mass is distributed across SOCIAL signals. `author_influence_score`
# is a social-GRAPH signal (PageRank over the user interaction graph) and is a
# first-class member of the default social ranker, not an optional extra.
DEFAULT_SOCIAL_WEIGHTS = {
    "topical_score": 0.50,
    "community_score": 0.14,
    "credibility_score": 0.08,
    "author_influence_score": 0.07,
    "engagement_score": 0.06,
    "freshness_score": 0.05,
    "tag_match_score": 0.05,
    "sentiment_alignment_score": 0.05,
}

# Intent profiles change the emphasis of the SAME signals for different
# information needs (these are query-intent presets, applied even without a
# searcher identity).
PROFILE_WEIGHTS = {
    "default": DEFAULT_SOCIAL_WEIGHTS,
    "beginner": {
        "topical_score": 0.44,
        "community_score": 0.20,
        "credibility_score": 0.08,
        "author_influence_score": 0.06,
        "engagement_score": 0.06,
        "freshness_score": 0.04,
        "tag_match_score": 0.04,
        "sentiment_alignment_score": 0.04,
        "clarity_score": 0.04,
    },
    "advanced_investor": {
        "topical_score": 0.48,
        "community_score": 0.10,
        "credibility_score": 0.12,
        "author_influence_score": 0.08,
        "engagement_score": 0.05,
        "freshness_score": 0.05,
        "tag_match_score": 0.05,
        "sentiment_alignment_score": 0.03,
        "technical_score": 0.04,
    },
    "risk_sensitive": {
        "topical_score": 0.46,
        "community_score": 0.12,
        "credibility_score": 0.13,
        "author_influence_score": 0.06,
        "engagement_score": 0.04,
        "freshness_score": 0.05,
        "tag_match_score": 0.04,
        "sentiment_alignment_score": 0.06,
        "risk_warning_score": 0.04,
    },
}

# --------------------------------------------------------------------------- #
# Searcher (social) profiles
# --------------------------------------------------------------------------- #
# A SEARCHER is an identity in the social network. The same query produces
# different rankings for different searchers because the re-ranker rewards
# content authored by users who are socially close to the searcher
# (`social_proximity_score`) and content from the searcher's interest
# communities (`community_affinity_score`). This is the defining property of
# social search (Viviani, slide "Social search as personalized search").
#
# `follow_tags` is used to RESOLVE, at runtime, which authors a searcher
# follows: the searcher is connected to the most influential authors whose
# expertise (dominant tags) matches their interests. This avoids hardcoding
# user ids and works on any corpus, real or sample.
SEARCHER_PROFILES = {
    "beginner_saver": {
        "intent_profile": "beginner",
        "follow_tags": ["saving", "budgeting", "emergency-fund", "debt", "credit-score"],
        "n_follow": 3,
    },
    "active_investor": {
        "intent_profile": "advanced_investor",
        "follow_tags": ["investing", "etf", "stocks", "tax", "retirement"],
        "n_follow": 3,
    },
    "cautious_planner": {
        "intent_profile": "risk_sensitive",
        "follow_tags": ["risk", "scam", "fraud", "mortgage", "tax"],
        "n_follow": 3,
    },
}

# Extra weight mass added when a searcher identity is supplied. These are the
# personalisation-by-social-graph signals.
SEARCHER_WEIGHTS = {
    "social_proximity_score": 0.13,
    "community_affinity_score": 0.07,
}

FINANCIAL_ASPECTS = [
    "debt",
    "tax",
    "investing",
    "retirement",
    "mortgage",
    "credit",
    "saving",
    "risk",
]


@dataclass(frozen=True)
class StackExchangePaths:
    """Configurable Stack Exchange dump file locations."""

    posts_xml: Path = RAW_DIR / "Posts.xml"
    users_xml: Path = RAW_DIR / "Users.xml"
    comments_xml: Path = RAW_DIR / "Comments.xml"
    votes_xml: Path = RAW_DIR / "Votes.xml"


def ensure_directories() -> None:
    """Create project output directories if they do not exist."""
    for path in [
        RAW_DIR,
        PROCESSED_DIR,
        QRELS_DIR,
        SAMPLE_DIR,
        RUNS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        REPORT_TABLES_DIR,
        REPORT_FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
