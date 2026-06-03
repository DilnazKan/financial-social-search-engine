"""Classical retrieval baselines: BM25 and TF-IDF."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import tokenize


@dataclass
class BM25Retriever:
    """Small pure-Python BM25 implementation for laptop-friendly retrieval."""

    documents: pd.DataFrame
    text_column: str = "combined_text"
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_tokens = [tokenize(text) for text in self.documents[self.text_column].fillna("")]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 1.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq = Counter()
        for freqs in self.term_freqs:
            self.doc_freq.update(freqs.keys())

    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        scores = np.zeros(len(self.documents), dtype=float)
        n_docs = len(self.documents)
        for term in terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for i, freqs in enumerate(self.term_freqs):
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1) / denom)
        return scores

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        scores = self.score(query)
        return _ranked_results(self.documents, scores, "bm25", top_k)


@dataclass
class TFIDFRetriever:
    """TF-IDF cosine similarity baseline."""

    documents: pd.DataFrame
    text_column: str = "combined_text"

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.documents[self.text_column].fillna(""))

    def score(self, query: str) -> np.ndarray:
        query_vec = self.vectorizer.transform([query])
        return cosine_similarity(query_vec, self.matrix).ravel()

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        scores = self.score(query)
        return _ranked_results(self.documents, scores, "tfidf", top_k)


def _ranked_results(docs: pd.DataFrame, scores: np.ndarray, run_name: str, top_k: int) -> pd.DataFrame:
    order = np.argsort(-scores)[:top_k]
    rows = docs.iloc[order].copy()
    rows["retrieval_score"] = scores[order]
    rows["rank"] = range(1, len(rows) + 1)
    rows["run"] = run_name
    return rows.reset_index(drop=True)


def retrieve_candidates(docs: pd.DataFrame, query: str, method: str = "bm25", candidates: int = 100) -> pd.DataFrame:
    """Retrieve candidate documents for social reranking."""
    if method.lower() == "tfidf":
        return TFIDFRetriever(docs).search(query, candidates)
    return BM25Retriever(docs).search(query, candidates)

