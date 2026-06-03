"""Generate report-ready visualizations for the social-search project."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from itertools import combinations

import pandas as pd

from . import config

try:
    mpl_config_dir = config.RESULTS_DIR / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
except Exception:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="financial-social-mpl-"))

import matplotlib.pyplot as plt
import networkx as nx

from .evaluation import run_evaluation
from .preprocessing import load_processed_documents, tokenize
from .retrieval import retrieve_candidates
from .reranking import apply_social_reranking
from .sentiment import financial_sentiment
from .social_graph import build_social_graph, resolve_searcher, influence_table

DEMO_QUERY = "Should I pay off credit card debt before investing?"


def truncate(value: object, max_chars: int = 76) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def savefig(name: str) -> None:
    config.ensure_directories()
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / name, dpi=180)
    plt.savefig(config.REPORT_FIGURES_DIR / name, dpi=180)
    plt.close()


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    labels = ["Query", "BM25 / TF-IDF\n(topical)", "Candidates", "Social features\n+ social graph",
              "Social re-ranker\n(+ searcher)", "Personalised\nresults"]
    xs = [0.06, 0.23, 0.40, 0.58, 0.77, 0.94]
    for x, label in zip(xs, labels):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="#d6e7e3", ec="#106b68"))
    for a, b in zip(xs, xs[1:]):
        ax.annotate("", xy=(b - 0.07, 0.5), xytext=(a + 0.07, 0.5), arrowprops=dict(arrowstyle="->", color="#172a31"))
    savefig("pipeline_diagram.png")


def before_after_example(docs: pd.DataFrame, graph_ctx) -> None:
    query = DEMO_QUERY
    bm25 = retrieve_candidates(docs, query, "bm25", 10)
    social = apply_social_reranking(retrieve_candidates(docs, query, "bm25", 100), query, graph_ctx=graph_ctx).head(10)
    bm25_rank_by_doc = dict(zip(bm25["doc_id"].astype(str), bm25["rank"]))
    rows = []
    for rank in range(1, 11):
        bm25_doc = bm25.iloc[rank - 1] if rank <= len(bm25) else pd.Series(dtype=object)
        social_doc = social.iloc[rank - 1] if rank <= len(social) else pd.Series(dtype=object)
        sid = str(social_doc.get("doc_id", ""))
        prev = bm25_rank_by_doc.get(sid)
        movement = "entered top 10" if prev is None else ("unchanged" if int(prev) == rank else f"moved {'up' if int(prev)-rank>0 else 'down'} {abs(int(prev)-rank)}")
        rows.append({
            "rank": rank,
            "bm25_doc_id": bm25_doc.get("doc_id", ""), "bm25_title": truncate(bm25_doc.get("title", "")),
            "social_doc_id": social_doc.get("doc_id", ""), "social_title": truncate(social_doc.get("title", "")),
            "social_final_score": round(float(social_doc.get("final_score", 0.0)), 4),
            "social_topical_score": round(float(social_doc.get("topical_score", 0.0)), 4),
            "social_community_score": round(float(social_doc.get("community_score", 0.0)), 4),
            "social_author_influence_score": round(float(social_doc.get("author_influence_score", 0.0)), 4),
            "social_credibility_score": round(float(social_doc.get("credibility_score", 0.0)), 4),
            "tags": social_doc.get("tags", ""), "what_changed": movement,
        })
    table = pd.DataFrame(rows)
    table.to_csv(config.FIGURES_DIR / "before_after_ranking_example.csv", index=False)
    table.to_csv(config.REPORT_TABLES_DIR / "before_after_ranking_example.csv", index=False)

    plot_rows = social[["doc_id", "title", "rank"]].copy()
    plot_rows["bm25_rank"] = plot_rows["doc_id"].astype(str).map(bm25_rank_by_doc).fillna(11)
    plot_rows = plot_rows.rename(columns={"rank": "social_rank"}).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    for y, (_, row) in zip(range(len(plot_rows)), plot_rows.iterrows()):
        ax.plot([row["bm25_rank"], row["social_rank"]], [y, y], color="#c5d5d1", linewidth=2)
        ax.scatter(row["bm25_rank"], y, color="#6c7a89", label="BM25 rank" if y == 0 else "")
        ax.scatter(row["social_rank"], y, color="#106b68", label="Social rank" if y == 0 else "")
    ax.set_yticks(list(range(len(plot_rows))))
    ax.set_yticklabels([f"{truncate(t,40)}" for t in plot_rows["title"]], fontsize=8)
    ax.set_xlim(11.5, 0.5)
    ax.set_xlabel("Rank position (left is better)")
    ax.set_title("Before vs after: BM25 candidates re-ranked with social signals")
    ax.legend(loc="lower right")
    savefig("before_after_ranking_example.png")


def feature_contribution(docs: pd.DataFrame, graph_ctx) -> None:
    social = apply_social_reranking(retrieve_candidates(docs, DEMO_QUERY, "bm25", 100), DEMO_QUERY, graph_ctx=graph_ctx).head(5)
    features = ["topical_score", "community_score", "credibility_score", "author_influence_score",
                "engagement_score", "freshness_score", "tag_match_score", "sentiment_alignment_score"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(social)
    for feature in features:
        values = (social[feature] * config.DEFAULT_SOCIAL_WEIGHTS.get(feature, 0.0)).tolist()
        ax.barh(social["doc_id"].astype(str), values, left=bottom, label=feature.replace("_score", ""))
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_title("Weighted feature contribution to final score (top 5)")
    ax.set_xlabel("Contribution to final score")
    ax.invert_yaxis()
    ax.legend(fontsize=8, ncols=2)
    savefig("social_feature_contribution.png")


def topic_network(docs: pd.DataFrame) -> None:
    tag_counts, edge_counts = Counter(), Counter()
    for tags in docs["tags"].fillna(""):
        unique = sorted(set(tokenize(tags)))
        tag_counts.update(unique)
        edge_counts.update(combinations(unique, 2))
    graph = nx.Graph()
    for tag, count in tag_counts.most_common(25):
        graph.add_node(tag, size=count)
    for (a, b), count in edge_counts.items():
        if a in graph and b in graph:
            graph.add_edge(a, b, weight=count)
    pos = nx.spring_layout(graph, seed=7)
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=[300 + graph.nodes[n]["size"] * 90 for n in graph], node_color="#106b68", alpha=0.85)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
    ax.set_title("Financial tag co-occurrence network")
    ax.axis("off")
    savefig("topic_network.png")


def social_graph_figure(graph_ctx) -> None:
    """Draw the REAL user interaction graph: node size = influence, colour = community."""
    g = graph_ctx.graph
    if g.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(g, seed=11, k=0.6)
    influence = graph_ctx.influence
    community = graph_ctx.community
    sizes = [120 + 1400 * influence.get(n, 0.0) for n in g.nodes]
    colors = [community.get(n, 0) for n in g.nodes]
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.2, arrows=True, arrowsize=7)
    nodes = nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=colors, cmap="tab10", alpha=0.9)
    # label only the most influential users
    top = sorted(influence.items(), key=lambda kv: kv[1], reverse=True)[:8]
    nx.draw_networkx_labels(g, pos, labels={n: str(n) for n, _ in top}, ax=ax, font_size=8)
    ax.set_title("User interaction graph: author influence (size = PageRank) and communities (colour)")
    ax.axis("off")
    savefig("user_interaction_or_author_influence.png")
    influence_table(graph_ctx).to_csv(config.REPORT_TABLES_DIR / "author_influence_table.csv", index=False)


def searcher_personalization(docs: pd.DataFrame, graph_ctx) -> None:
    """Show that the SAME query is ranked differently for different searchers."""
    query = "How should I start investing?"
    cand = retrieve_candidates(docs, query, "bm25", 100)
    panels = {"no searcher": apply_social_reranking(cand, query, graph_ctx=graph_ctx).head(6)}
    for name in ["active_investor", "cautious_planner"]:
        searcher = resolve_searcher(graph_ctx, config.SEARCHER_PROFILES[name], name)
        panels[name] = apply_social_reranking(cand, query, graph_ctx=graph_ctx, searcher=searcher).head(6)

    # report-ready table: rank of each doc under each searcher
    all_docs = []
    for frame in panels.values():
        all_docs.extend(frame["doc_id"].astype(str).tolist())
    all_docs = list(dict.fromkeys(all_docs))
    rows = []
    for did in all_docs:
        row = {"doc_id": did}
        for name, frame in panels.items():
            m = frame[frame["doc_id"].astype(str) == did]
            row[f"rank_{name.replace(' ', '_')}"] = int(m["rank"].iloc[0]) if len(m) else None
        title = docs[docs["doc_id"].astype(str) == did]["title"]
        row["title"] = truncate(title.iloc[0], 50) if len(title) else ""
        row["author"] = int(docs[docs["doc_id"].astype(str) == did]["owner_user_id"].iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(config.REPORT_TABLES_DIR / "searcher_personalization_example.csv", index=False)

    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.5), sharey=False)
    for ax, (name, frame) in zip(axes, panels.items()):
        ax.barh(frame["doc_id"].astype(str), frame["final_score"], color="#106b68", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("final score")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"Same query, different searchers: '{query}'", fontsize=12)
    savefig("searcher_personalization_example.png")


def aspect_sentiment(docs: pd.DataFrame) -> None:
    rows = []
    for _, row in docs.iterrows():
        text = f"{row.get('tags', '')} {row.get('combined_text', '')}"
        aspect = row.get("aspect") or next((a for a in config.FINANCIAL_ASPECTS if a in text), "general")
        rows.append({"aspect": aspect, "sentiment": financial_sentiment(text)})
    counts = pd.DataFrame(rows).value_counts(["aspect", "sentiment"]).unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(9, 4))
    plt.title("Aspect-level financial sentiment")
    plt.xlabel("Aspect")
    plt.ylabel("Documents")
    savefig("aspect_sentiment_chart.png")


def evaluation_charts():
    full, ablation = run_evaluation()
    # Grouped bars: nDCG@10 by run across the three gold standards.
    pivot = full.pivot(index="run", columns="gold", values="nDCG@10")
    pivot = pivot[["topical", "social", "combined"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#6c7a89", "#d6654e", "#106b68"])
    ax.set_title("nDCG@10 by system and gold standard")
    ax.set_ylabel("nDCG@10")
    ax.set_xlabel("")
    ax.legend(title="gold standard")
    ax.tick_params(axis="x", rotation=0)
    savefig("evaluation_metrics.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ablation.plot(x="run", y="nDCG@10", kind="bar", ax=ax, color="#d6654e", legend=False)
    full_v = float(ablation.loc[ablation["run"] == "full_social", "nDCG@10"].iloc[0])
    ax.axhline(full_v, color="#106b68", linestyle="--", label="full social")
    ax.set_title("Ablation (combined gold): effect of removing each social feature")
    ax.set_ylabel("nDCG@10")
    ax.set_xlabel("")
    ax.legend()
    plt.xticks(rotation=30, ha="right", fontsize=8)
    savefig("ablation_chart.png")
    return full, ablation


def report_ready_tables(full: pd.DataFrame, ablation: pd.DataFrame, graph_ctx) -> None:
    config.ensure_directories()

    feature_rows = [
        ("topical_score", "BM25 / TF-IDF retrieval score", "Anchors ranking to the information need", "Answers must be on-topic before social evidence helps"),
        ("community_score", "answer_score + accepted_answer", "Collective approval / accepted usefulness", "Core social signal from the Q&A community"),
        ("engagement_score", "comments, question views, answer count", "Discussion activity and attention", "Promotes answers on questions that drew interaction"),
        ("credibility_score", "log1p(author_reputation)", "Authority matters for financial advice", "Lightweight author authority signal"),
        ("author_influence_score", "PageRank over the user interaction graph", "Network influence beyond raw reputation", "SOCIAL-GRAPH signal: influential community members"),
        ("freshness_score", "exp(-age_days / half_life)", "Tax/rates/products go out of date", "Temporal social relevance"),
        ("tag_match_score", "Jaccard(query terms, community tags)", "Community topic interpretation", "Connects user wording to socially assigned topics"),
        ("sentiment_alignment_score", "query intent + financial stance", "Cautious answers beat positive ones for risk/scam/debt/tax", "Stance usefulness, not positivity"),
        ("social_proximity_score", "graph distance to searcher's followed authors", "Content from the searcher's network", "SOCIAL-GRAPH personalisation (per searcher)"),
        ("community_affinity_score", "overlap with searcher's interest communities", "Shared-interest communities", "SOCIAL-GRAPH personalisation (per searcher)"),
    ]
    pd.DataFrame(feature_rows, columns=["feature", "computed_from", "why_it_matters", "social_search_role"]).to_csv(
        config.REPORT_TABLES_DIR / "social_features_table.csv", index=False)

    weight_rows = [{"feature": f, "default_weight": w} for f, w in config.DEFAULT_SOCIAL_WEIGHTS.items()]
    weight_rows += [{"feature": f + " (searcher only)", "default_weight": w} for f, w in config.SEARCHER_WEIGHTS.items()]
    pd.DataFrame(weight_rows).to_csv(config.REPORT_TABLES_DIR / "ranking_formula_weights.csv", index=False)

    # Mapping to the professor's ACTUAL 8-point rubric.
    rubric_rows = [
        ("Objective: completeness & clarity", "1",
         "README objective + 'why this is social search'; domain = personal-finance Q&A social search.",
         "README.md, report_assets/presentation_outline_15min.md"),
        ("Data: correctness", "1",
         "Real Money Stack Exchange dump loader (asker, answerer, commenters, votes, reputation) + reproducible offline corpus.",
         "src/data_collection.py, data/processed/, data/sample/"),
        ("Models: correctness", "2.5",
         "BM25 + TF-IDF baselines; transparent social re-ranker; SOCIAL GRAPH (PageRank influence, community detection) and searcher personalisation.",
         "src/retrieval.py, src/social_features.py, src/social_graph.py, src/reranking.py"),
        ("Evaluation: metrics correctness", "2.5",
         "P@k, R@k, MAP, nDCG@k; model-INDEPENDENT dual gold (topical + social usefulness) with TREC-style pooling; ablation.",
         "src/evaluation.py, results/metrics/evaluation_dual_gold.csv, results/metrics/ablation_metrics.csv"),
        ("Evaluation: clarity of discussion", "1",
         "Auto-generated discussion incl. anti-circularity note; human-annotation template provided for the definitive run.",
         "results/metrics/evaluation_discussion.md, data/qrels/social_judgments_template.csv"),
    ]
    pd.DataFrame(rubric_rows, columns=["rubric_item", "max_points", "repository_evidence", "main_files"]).to_csv(
        config.REPORT_TABLES_DIR / "grading_requirements.csv", index=False)

    full.to_csv(config.REPORT_TABLES_DIR / "evaluation_summary.csv", index=False)
    ablation.to_csv(config.REPORT_TABLES_DIR / "ablation_summary.csv", index=False)

    stats = pd.DataFrame([{
        "users": graph_ctx.graph.number_of_nodes(),
        "interaction_edges": graph_ctx.graph.number_of_edges(),
        "communities": len(set(graph_ctx.community.values())),
        "most_influential_user": max(graph_ctx.influence, key=graph_ctx.influence.get) if graph_ctx.influence else None,
    }])
    stats.to_csv(config.REPORT_TABLES_DIR / "social_graph_stats.csv", index=False)


def main() -> None:
    docs = load_processed_documents()
    graph_ctx = build_social_graph(docs)
    pipeline_diagram()
    before_after_example(docs, graph_ctx)
    feature_contribution(docs, graph_ctx)
    topic_network(docs)
    social_graph_figure(graph_ctx)
    searcher_personalization(docs, graph_ctx)
    aspect_sentiment(docs)
    full, ablation = evaluation_charts()
    report_ready_tables(full, ablation, graph_ctx)
    print(f"Figures written to {config.FIGURES_DIR}")
    print(f"Report tables written to {config.REPORT_TABLES_DIR}")


if __name__ == "__main__":
    main()
