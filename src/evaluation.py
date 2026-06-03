"""Evaluation for financial social search: model-INDEPENDENT, dual ground truth.

WHY THIS DESIGN
---------------
A social search engine must be judged on whether it surfaces SOCIALLY USEFUL
content, not only topically matching text. The relevance judgments here are:

  * built from a transparent, documented rubric;
  * computed ONCE from query/document content and real community signals,
    BEFORE any ranking -- they never depend on a system's own scores or ranks
    (this is the key property that removes circularity);
  * provided in TWO complementary gold standards plus a holistic combined one:
      - topical_gold  : independent topical relevance (title+tags+aspect overlap).
                        Fairness check that social re-ranking does not damage
                        topical relevance.
      - social_gold   : community-endorsed usefulness (accepted + votes +
                        reputation). The contribution: does social re-ranking
                        surface community-validated answers better?
      - combined_gold : a holistic annotator (useful => must be on-topic).

We are explicit that community signals appear both as ranking FEATURES and, on
real data, inside the usefulness gold; the definitive evaluation therefore uses
the human-annotation template (provided in data/qrels/), which these reference
judgments illustrate but do not replace.

Pooling follows TREC practice: the judged pool per query is the union of the
top-k of every system, so no single system's ranking biases the label set.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .preprocessing import load_processed_documents, tokenize
from .retrieval import retrieve_candidates
from .reranking import apply_social_reranking
from .social_graph import build_social_graph


# (query, aspect) -- natural-language queries mapped to a financial aspect so the
# independent topical gold can use aspect agreement as one signal.
EVAL_QUERIES = [
    ("Should I pay off credit card debt before investing?", "debt"),
    ("Fastest way to get out of credit card debt", "debt"),
    ("How big should my emergency fund be?", "saving"),
    ("Where is the safest place to keep emergency savings?", "saving"),
    ("Is renting better than buying a home?", "mortgage"),
    ("When does refinancing a mortgage make sense?", "mortgage"),
    ("How are long term capital gains taxed?", "tax"),
    ("What paperwork do I need for investment taxes?", "tax"),
    ("Are ETFs safer than picking individual stocks?", "investing"),
    ("Lump sum investing versus dollar cost averaging", "investing"),
    ("Which fees should I check before buying an ETF?", "investing"),
    ("I think an investment might be a scam, what should I do?", "risk"),
    ("How can I avoid investment fraud?", "risk"),
    ("Roth versus traditional retirement account", "retirement"),
    ("How do I raise my credit score?", "credit"),
    ("What credit card utilisation is healthy?", "credit"),
    ("Should I hold cash or short term bonds for savings?", "saving"),
    ("Should I overpay my mortgage or invest the money?", "mortgage"),
]


def ensure_queries() -> pd.DataFrame:
    config.ensure_directories()
    rows = [{"query_id": f"Q{i:02d}", "query": q, "aspect": a} for i, (q, a) in enumerate(EVAL_QUERIES, start=1)]
    df = pd.DataFrame(rows)
    df.to_csv(config.SOCIAL_QUERIES_CSV, index=False)
    return df


# --------------------------------------------------------------------------- #
# Metric implementations (graded relevance, threshold-based binary metrics)
# --------------------------------------------------------------------------- #
def dcg(labels: list[float], k: int) -> float:
    return sum((2 ** label - 1) / np.log2(i + 2) for i, label in enumerate(labels[:k]))


def ndcg_at_k(ranked_doc_ids: list[str], labels: dict[str, float], k: int) -> float:
    gains = [labels.get(str(doc_id), 0.0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal, k)
    return dcg(gains, k) / ideal_dcg if ideal_dcg else 0.0


def precision_at_k(ranked_doc_ids: list[str], labels: dict[str, float], k: int, threshold: int = 2) -> float:
    if not ranked_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if labels.get(str(doc_id), 0) >= threshold)
    return hits / min(k, len(ranked_doc_ids))


def average_precision(ranked_doc_ids: list[str], labels: dict[str, float], threshold: int = 2) -> float:
    relevant = {doc for doc, label in labels.items() if label >= threshold}
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked_doc_ids, start=1):
        if str(doc_id) in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def recall_at_k(ranked_doc_ids: list[str], labels: dict[str, float], k: int, threshold: int = 2) -> float:
    relevant = {doc for doc, label in labels.items() if label >= threshold}
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if str(doc_id) in relevant)
    return hits / len(relevant)


# --------------------------------------------------------------------------- #
# Model-independent relevance judgments
# --------------------------------------------------------------------------- #
def _bucket(value: float, edges=(0.15, 0.30, 0.50)) -> int:
    return sum(value >= e for e in edges)  # -> 0,1,2,3


def _topical_label(query: str, aspect: str, row: pd.Series) -> int:
    """Independent topical relevance from title+tags overlap and aspect match.

    Uses metadata fields (title, tags, aspect), NOT the combined_text that the
    rankers score and NOT any system score -- so it is independent of the
    systems under test.
    """
    q_terms = set(tokenize(query))
    basis = set(tokenize(str(row.get("title", "")))) | set(tokenize(str(row.get("tags", ""))))
    if not q_terms or not basis:
        overlap = 0.0
    else:
        overlap = len(q_terms & basis) / len(q_terms | basis)
    label = _bucket(overlap)
    doc_aspect = str(row.get("aspect", "")).lower()
    if doc_aspect and doc_aspect == aspect.lower():
        label = max(label, 2)          # on-aspect answers are at least marginally relevant
    elif doc_aspect and doc_aspect != aspect.lower():
        label = min(label, 1)          # off-aspect answers capped low
    return int(label)


def _social_usefulness(pool: pd.DataFrame) -> dict[str, int]:
    """Community-endorsed usefulness per doc, normalised within the query pool."""
    score = pd.to_numeric(pool.get("answer_score", 0), errors="coerce").fillna(0).astype(float)
    rep = np.log1p(pd.to_numeric(pool.get("author_reputation", 0), errors="coerce").fillna(0).astype(float))
    accepted = pool.get("accepted_answer", False).astype(bool).astype(float)

    def norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(np.zeros(len(s)), index=s.index)

    usefulness = 0.55 * norm(score) + 0.25 * accepted + 0.20 * norm(rep)
    return {str(d): _bucket(v, edges=(0.20, 0.45, 0.70)) for d, v in zip(pool["doc_id"], usefulness)}


def build_relevance_judgments(docs: pd.DataFrame, graph_ctx, top_k: int = 10) -> pd.DataFrame:
    """Build pooled, model-independent topical / social / combined judgments."""
    queries = ensure_queries()
    records = []
    for _, qrow in queries.iterrows():
        query, aspect = qrow["query"], qrow["aspect"]
        # Pool = union of all systems' top-k (TREC-style pooling).
        pool_ids: list[str] = []
        bm25 = retrieve_candidates(docs, query, "bm25", top_k)
        tfidf = retrieve_candidates(docs, query, "tfidf", top_k)
        social = apply_social_reranking(
            retrieve_candidates(docs, query, "bm25", config.DEFAULT_CANDIDATES),
            query, "default", graph_ctx=graph_ctx,
        ).head(top_k)
        for frame in (bm25, tfidf, social):
            pool_ids.extend(frame["doc_id"].astype(str).tolist())
        pool_ids = list(dict.fromkeys(pool_ids))
        pool = docs[docs["doc_id"].astype(str).isin(pool_ids)].copy()

        social_lbl = _social_usefulness(pool)
        for _, drow in pool.iterrows():
            did = str(drow["doc_id"])
            topical = _topical_label(query, aspect, drow)
            social_use = social_lbl.get(did, 0)
            combined = int(round(0.5 * topical + 0.5 * social_use)) if topical >= 1 else 0
            records.append(
                {
                    "query_id": qrow["query_id"],
                    "query": query,
                    "aspect": aspect,
                    "doc_id": did,
                    "topical_gold_0_3": topical,
                    "social_gold_0_3": social_use,
                    "combined_gold_0_3": combined,
                }
            )
    judged = pd.DataFrame(records)
    judged.to_csv(config.REFERENCE_JUDGMENTS_CSV, index=False)
    _write_human_template(judged)
    return judged


def _write_human_template(judged: pd.DataFrame) -> None:
    """Write an EMPTY human-annotation template (the gold for the final report)."""
    template = judged[["query_id", "query", "doc_id"]].copy()
    for col in ["topical_relevance_0_3", "social_usefulness_0_3", "credibility_0_3",
                "freshness_0_3", "final_manual_label_0_3", "notes"]:
        template[col] = ""
    template.to_csv(config.JUDGMENT_TEMPLATE_CSV, index=False)


# --------------------------------------------------------------------------- #
# Runs and evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(ranking_by_q: dict[str, list[str]], judged: pd.DataFrame, label_col: str, run_name: str) -> dict:
    metrics = {"run": run_name, "gold": label_col.replace("_gold_0_3", "")}
    acc = {"Precision@5": [], "Precision@10": [], "Recall@10": [], "MAP": [], "nDCG@5": [], "nDCG@10": []}
    for qid, group in judged.groupby("query_id"):
        labels = dict(zip(group["doc_id"].astype(str), pd.to_numeric(group[label_col], errors="coerce").fillna(0)))
        ranking = [str(x) for x in ranking_by_q.get(qid, [])]
        acc["Precision@5"].append(precision_at_k(ranking, labels, 5))
        acc["Precision@10"].append(precision_at_k(ranking, labels, 10))
        acc["Recall@10"].append(recall_at_k(ranking, labels, 10))
        acc["MAP"].append(average_precision(ranking, labels))
        acc["nDCG@5"].append(ndcg_at_k(ranking, labels, 5))
        acc["nDCG@10"].append(ndcg_at_k(ranking, labels, 10))
    metrics.update({k: float(np.mean(v)) for k, v in acc.items()})
    return metrics


def build_runs(docs: pd.DataFrame, judged: pd.DataFrame, graph_ctx) -> dict[str, dict[str, list[str]]]:
    queries = judged[["query_id", "query"]].drop_duplicates()
    runs = {"BM25": {}, "TF-IDF": {}, "BM25 + Social": {}}
    for _, row in queries.iterrows():
        qid, query = row["query_id"], row["query"]
        runs["BM25"][qid] = retrieve_candidates(docs, query, "bm25", 10)["doc_id"].astype(str).tolist()
        runs["TF-IDF"][qid] = retrieve_candidates(docs, query, "tfidf", 10)["doc_id"].astype(str).tolist()
        social = apply_social_reranking(
            retrieve_candidates(docs, query, "bm25", config.DEFAULT_CANDIDATES),
            query, "default", graph_ctx=graph_ctx,
        ).head(10)
        runs["BM25 + Social"][qid] = social["doc_id"].astype(str).tolist()
        social.to_csv(config.RUNS_DIR / f"{qid}_social.csv", index=False)
    return runs


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_directories()
    docs = load_processed_documents()
    graph_ctx = build_social_graph(docs)
    judged = build_relevance_judgments(docs, graph_ctx)
    runs = build_runs(docs, judged, graph_ctx)

    # Evaluate every system against every gold standard.
    rows = []
    for gold in ["topical_gold_0_3", "social_gold_0_3", "combined_gold_0_3"]:
        for name, ranking in runs.items():
            rows.append(evaluate_run(ranking, judged, gold, name))
    full = pd.DataFrame(rows)
    full.to_csv(config.METRICS_DIR / "evaluation_dual_gold.csv", index=False)

    # Backward-compatible split files (primary gold = combined).
    primary = full[full["gold"] == "combined"]
    primary[primary["run"] != "BM25 + Social"].to_csv(config.METRICS_DIR / "baseline_metrics.csv", index=False)
    primary[primary["run"] == "BM25 + Social"].to_csv(config.METRICS_DIR / "social_metrics.csv", index=False)

    # Ablation against the combined gold.
    ablations = []
    queries = judged[["query_id", "query"]].drop_duplicates()
    for ablate in [None, "community_score", "credibility_score", "author_influence_score",
                   "freshness_score", "engagement_score", "sentiment_alignment_score"]:
        run = {}
        for _, row in queries.iterrows():
            cand = retrieve_candidates(docs, row["query"], "bm25", config.DEFAULT_CANDIDATES)
            reranked = apply_social_reranking(cand, row["query"], "default", ablate=ablate, graph_ctx=graph_ctx).head(10)
            run[row["query_id"]] = reranked["doc_id"].astype(str).tolist()
        name = "full_social" if ablate is None else f"without_{ablate}"
        ablations.append(evaluate_run(run, judged, "combined_gold_0_3", name))
    ablation_df = pd.DataFrame(ablations)
    ablation_df.to_csv(config.METRICS_DIR / "ablation_metrics.csv", index=False)

    _write_discussion(full, ablation_df, len(docs), len(queries), graph_ctx)
    return full, ablation_df


def _write_discussion(full: pd.DataFrame, ablation: pd.DataFrame, n_docs: int, n_queries: int, graph_ctx) -> None:
    def get(run, gold, metric):
        sub = full[(full["run"] == run) & (full["gold"] == gold)]
        return float(sub[metric].iloc[0]) if len(sub) else float("nan")

    soc_combined = get("BM25 + Social", "combined", "nDCG@10")
    bm_combined = get("BM25", "combined", "nDCG@10")
    soc_topical = get("BM25 + Social", "topical", "nDCG@10")
    bm_topical = get("BM25", "topical", "nDCG@10")
    soc_social = get("BM25 + Social", "social", "nDCG@10")
    bm_social = get("BM25", "social", "nDCG@10")

    lines = [
        "# Evaluation discussion (auto-generated)\n",
        f"Corpus: {n_docs} answer documents. Queries: {n_queries}. "
        f"Social graph: {graph_ctx.graph.number_of_nodes()} users, "
        f"{graph_ctx.graph.number_of_edges()} interaction edges, "
        f"{len(set(graph_ctx.community.values()))} detected communities.\n",
        "## Headline numbers (nDCG@10)\n",
        f"- Social-usefulness gold: BM25 = {bm_social:.3f} vs BM25+Social = {soc_social:.3f} "
        f"(delta {soc_social - bm_social:+.3f}).",
        f"- Topical gold (fairness check): BM25 = {bm_topical:.3f} vs BM25+Social = {soc_topical:.3f} "
        f"(delta {soc_topical - bm_topical:+.3f}).",
        f"- Combined gold: BM25 = {bm_combined:.3f} vs BM25+Social = {soc_combined:.3f} "
        f"(delta {soc_combined - bm_combined:+.3f}).\n",
        "## How to read this\n",
        "The social re-ranker is expected to help most on the SOCIAL-usefulness gold, "
        "because it incorporates community approval, author credibility and graph influence: "
        "it promotes community-validated answers that a purely lexical ranker places lower. "
        "On the TOPICAL gold the two systems should be close, confirming that social "
        "re-ranking reorders within topically relevant candidates rather than sacrificing topicality.\n",
        "## Honesty / anti-circularity note\n",
        "Relevance labels are computed once, from query/document content (topical) and real "
        "community signals (social), and never depend on any system's own ranks or scores. "
        "Because community signals appear both as ranking features and inside the social gold, "
        "the social re-ranker has a built-in advantage on that gold; this is why we also report "
        "the independent topical gold and provide an empty human-annotation template "
        "(`data/qrels/social_judgments_template.csv`) for the definitive report evaluation.\n",
        "## Ablation (combined gold, nDCG@10)\n",
    ]
    for _, r in ablation.iterrows():
        lines.append(f"- {r['run']}: {r['nDCG@10']:.3f}")
    (config.METRICS_DIR / "evaluation_discussion.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10)
    parser.parse_args()
    full, ablation = run_evaluation()
    print("Evaluation complete (model-independent, dual gold).\n")
    print(full.to_string(index=False))
    print("\nAblation (combined gold):")
    print(ablation[["run", "nDCG@10", "MAP"]].to_string(index=False))


if __name__ == "__main__":
    main()
