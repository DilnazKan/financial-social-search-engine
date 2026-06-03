"""Lightweight financial sentiment and intent alignment fallback."""

from __future__ import annotations

from .preprocessing import tokenize


POSITIVE = {"save", "savings", "benefit", "diversified", "low-cost", "guaranteed", "improve", "liquid"}
NEGATIVE = {"debt", "scam", "fraud", "risk", "loss", "warning", "avoid", "volatile", "pressure", "interest"}
CAUTIOUS = {"depends", "consider", "compare", "consult", "verify", "risk", "cautious", "stress-test", "records"}

INTENT_KEYWORDS = {
    "risk_warning": {"scam", "fraud", "risk", "warning", "avoid", "safe"},
    "investing": {"invest", "stock", "etf", "portfolio", "market", "bond"},
    "tax": {"tax", "capital", "gain", "deduction", "basis"},
    "debt_credit": {"debt", "credit", "card", "loan", "score"},
    "saving_budgeting": {"emergency", "saving", "budget", "cash", "fund"},
}


def detect_query_intent(query: str) -> str:
    """Classify query into an explainable financial intent category."""
    terms = set(tokenize(query))
    for intent, keywords in INTENT_KEYWORDS.items():
        if terms & keywords:
            return intent
    return "general"


def financial_sentiment(text: str) -> str:
    """Rule-based fallback sentiment: positive, negative, cautious or neutral."""
    terms = set(tokenize(text))
    if len(terms & CAUTIOUS) >= 2:
        return "cautious"
    pos = len(terms & POSITIVE)
    neg = len(terms & NEGATIVE)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    if terms & CAUTIOUS:
        return "cautious"
    return "neutral"


def sentiment_alignment(query: str, text: str) -> float:
    """Score how useful the answer stance is for the query intent."""
    intent = detect_query_intent(query)
    sentiment = financial_sentiment(text)
    table = {
        "risk_warning": {"cautious": 1.0, "negative": 0.9, "neutral": 0.6, "positive": 0.2},
        "debt_credit": {"cautious": 0.9, "negative": 0.8, "neutral": 0.7, "positive": 0.5},
        "tax": {"neutral": 1.0, "cautious": 0.9, "negative": 0.5, "positive": 0.4},
        "investing": {"neutral": 0.8, "cautious": 0.9, "positive": 0.7, "negative": 0.5},
        "saving_budgeting": {"neutral": 0.8, "positive": 0.8, "cautious": 0.9, "negative": 0.5},
        "general": {"neutral": 0.8, "cautious": 0.8, "positive": 0.7, "negative": 0.6},
    }
    return table[intent].get(sentiment, 0.5)

