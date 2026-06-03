"""Command-line demo for Financial Social Search (graph- and searcher-aware)."""

from __future__ import annotations

import argparse
import textwrap

from . import config
from .preprocessing import load_processed_documents
from .retrieval import retrieve_candidates
from .reranking import apply_social_reranking
from .social_graph import build_social_graph, resolve_searcher


def snippet(text: str, width: int = 160) -> str:
    return textwrap.shorten(str(text).replace("\n", " "), width=width, placeholder="...")


def run_demo(query: str, profile: str = "default", searcher_name: str | None = None,
             top_k: int = 10, method: str = "bm25") -> None:
    docs = load_processed_documents()
    graph_ctx = build_social_graph(docs)

    searcher = None
    if searcher_name and searcher_name in config.SEARCHER_PROFILES:
        searcher = resolve_searcher(graph_ctx, config.SEARCHER_PROFILES[searcher_name], searcher_name)

    candidates = retrieve_candidates(docs, query, method=method, candidates=max(config.DEFAULT_CANDIDATES, top_k))
    reranked = apply_social_reranking(candidates, query, profile=profile, graph_ctx=graph_ctx, searcher=searcher).head(top_k)

    header = f"\nFinancial Social Search | query={query!r} | method={method}"
    if searcher:
        header += (f" | searcher={searcher.name} (follows authors {sorted(searcher.followed_authors)},"
                   f" communities {sorted(searcher.followed_communities)})")
    else:
        header += f" | profile={profile}"
    print(header + "\n")

    show_personal = searcher is not None
    for _, row in reranked.iterrows():
        print(f"{int(row['rank']):>2}. {row['title']}")
        print(f"    doc_id={row['doc_id']} author={row.get('owner_user_id')} tags={row.get('tags','')}")
        base = ("    final={:.3f} topical={:.3f} community={:.3f} credibility={:.3f} "
                "influence={:.3f} engagement={:.3f} freshness={:.3f} sentiment={:.3f}").format(
            row["final_score"], row["topical_score"], row["community_score"], row["credibility_score"],
            row.get("author_influence_score", 0.0), row["engagement_score"], row["freshness_score"],
            row["sentiment_alignment_score"])
        if show_personal:
            base += " proximity={:.3f} affinity={:.3f}".format(
                row.get("social_proximity_score", 0.0), row.get("community_affinity_score", 0.0))
        print(base)
        print(f"    {snippet(row['answer_body'])}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--profile", default="default", choices=list(config.PROFILE_WEIGHTS))
    parser.add_argument("--searcher", default=None, choices=list(config.SEARCHER_PROFILES),
                        help="social searcher identity; enables social-graph personalisation")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--method", default="bm25", choices=["bm25", "tfidf"])
    args = parser.parse_args()
    run_demo(args.query, args.profile, args.searcher, args.top_k, args.method)


if __name__ == "__main__":
    main()
