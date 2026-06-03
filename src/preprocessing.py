"""Preprocess Stack Exchange financial Q&A documents."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-']+")


def clean_html(value: object) -> str:
    """Strip HTML tags/entities and normalize whitespace."""
    text = "" if value is None or pd.isna(value) else str(value)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_tags(value: object) -> str:
    """Convert Stack Exchange '<tax><capital-gains>' tags to 'tax capital-gains'."""
    text = "" if value is None or pd.isna(value) else str(value)
    if "<" in text and ">" in text:
        return " ".join(re.findall(r"<([^>]+)>", text))
    return SPACE_RE.sub(" ", text.replace("|", " ")).strip()


def tokenize(text: object) -> list[str]:
    """Tokenize text for retrieval and feature matching."""
    return [token.lower() for token in TOKEN_RE.findall("" if text is None else str(text))]


def create_combined_text(row: pd.Series) -> str:
    """Build the searchable text field from title, question, answer and tags."""
    parts = [
        row.get("title", ""),
        row.get("question_body", ""),
        row.get("answer_body", ""),
        row.get("tags", ""),
    ]
    return SPACE_RE.sub(" ", " ".join(str(part) for part in parts if str(part).strip())).strip()


def clean_documents(df: pd.DataFrame, min_answer_chars: int = 40) -> pd.DataFrame:
    """Clean text fields, compute dates/age and remove empty answers."""
    docs = df.copy()
    for column in ["title", "question_body", "answer_body"]:
        if column not in docs:
            docs[column] = ""
        docs[column] = docs[column].map(clean_html)
    docs["tags"] = docs.get("tags", "").map(normalize_tags)
    docs["combined_text"] = docs.apply(create_combined_text, axis=1)
    docs = docs[docs["answer_body"].str.len() >= min_answer_chars].copy()

    for column in ["creation_date", "last_activity_date"]:
        docs[column] = pd.to_datetime(docs.get(column), errors="coerce", utc=True)
    reference_date = pd.Timestamp.utcnow()
    docs["last_activity_date"] = docs["last_activity_date"].fillna(docs["creation_date"])
    docs["creation_date"] = docs["creation_date"].fillna(docs["last_activity_date"])
    docs["age_days"] = (reference_date - docs["last_activity_date"]).dt.days.clip(lower=0).fillna(3650)

    numeric_defaults = {
        "answer_score": 0,
        "comment_count": 0,
        "question_view_count": 0,
        "answer_count": 0,
        "author_reputation": 0,
    }
    for column, default in numeric_defaults.items():
        docs[column] = pd.to_numeric(docs.get(column, default), errors="coerce").fillna(default)
    docs["accepted_answer"] = docs.get("accepted_answer", False).fillna(False).astype(bool)
    docs["doc_id"] = docs["doc_id"].astype(str)
    return docs.reset_index(drop=True)


def save_processed_documents(df: pd.DataFrame) -> None:
    """Save processed documents as CSV and Parquet when supported."""
    config.ensure_directories()
    df.to_csv(config.PROCESSED_CSV, index=False)
    df.head(min(25, len(df))).to_csv(config.SAMPLE_CSV, index=False)
    try:
        df.to_parquet(config.PROCESSED_PARQUET, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional pyarrow
        print(f"Parquet export skipped: {exc}")


def load_processed_documents(path: Path | None = None) -> pd.DataFrame:
    """Load processed documents, generating fallback sample data if needed."""
    target = path or config.PROCESSED_CSV
    if target.exists():
        return pd.read_csv(target)
    if config.SAMPLE_CSV.exists():
        print(f"Processed data missing; falling back to {config.SAMPLE_CSV}.")
        sample = pd.read_csv(config.SAMPLE_CSV)
        if "combined_text" not in sample.columns:
            sample = clean_documents(sample)
        return sample
    from .data_collection import build_sample_dataset

    print("Processed data missing; generating fallback sample dataset.")
    docs = clean_documents(build_sample_dataset())
    save_processed_documents(docs)
    return docs
