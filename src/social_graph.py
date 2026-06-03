"""Social-graph layer for financial social search.

The professor's framing of social search (Viviani, "Social Search" slides)
states that a social search engine must, beyond topical matching, consider:

  * the SOCIAL RELATIONSHIPS between the searcher and the results
    (friendships, follows, shared interests, network interactions);
  * INFLUENCE METRICS of the content authors;
  * SOCIAL-GRAPH INDEXING of the user interaction network.

This module builds that layer. From the corpus we construct a directed user
interaction graph (askers -> answerers, commenters -> authors), compute author
INFLUENCE via PageRank, detect interest COMMUNITIES via modularity, and expose
SEARCHER-aware features (social proximity and community affinity) so that the
same query can be ranked differently for different searchers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd

from .preprocessing import tokenize


@dataclass
class SocialGraphContext:
    """Pre-computed social-graph artefacts shared across queries."""

    graph: nx.DiGraph
    influence: dict[int, float] = field(default_factory=dict)        # user_id -> [0,1]
    community: dict[int, int] = field(default_factory=dict)          # user_id -> community id
    author_main_tags: dict[int, set[str]] = field(default_factory=dict)
    _undirected: nx.Graph | None = None

    def undirected(self) -> nx.Graph:
        if self._undirected is None:
            self._undirected = self.graph.to_undirected()
        return self._undirected


def build_social_graph(docs: pd.DataFrame, comment_edges: list[tuple[int, int]] | None = None) -> SocialGraphContext:
    """Build the user interaction graph and derive influence + communities.

    Edges (directed, "engaged-with"):
      asker  -> answerer   : the asker's question was answered by the answerer
      commenter -> author  : a commenter engaged with the author's answer

    PageRank on this graph rewards users who are consistently sought out to
    answer / are commented on, i.e. influential community members.
    """
    graph = nx.DiGraph()

    # Derive commenter -> author edges from the corpus column unless supplied.
    if comment_edges is None:
        comment_edges = []
        if "commenter_user_ids" in docs.columns:
            for _, row in docs.iterrows():
                author = _as_user(row.get("owner_user_id"))
                raw = str(row.get("commenter_user_ids", "") or "")
                for tok in raw.split():
                    commenter = _as_user(tok)
                    if commenter is not None and author is not None:
                        comment_edges.append((commenter, author))

    for _, row in docs.iterrows():
        answerer = _as_user(row.get("owner_user_id"))
        asker = _as_user(row.get("question_owner_user_id"))
        if answerer is None:
            continue
        graph.add_node(answerer)
        # collect the answerer's areas of expertise from the question tags
        tags = set(tokenize(str(row.get("tags", ""))))
        ctx_tags = graph.nodes[answerer].get("tags", set())
        graph.nodes[answerer]["tags"] = ctx_tags | tags
        if asker is not None and asker != answerer:
            graph.add_node(asker)
            if graph.has_edge(asker, answerer):
                graph[asker][answerer]["weight"] += 1
            else:
                graph.add_edge(asker, answerer, weight=1)

    for commenter, author in comment_edges or []:
        if commenter is None or author is None or commenter == author:
            continue
        graph.add_node(commenter)
        graph.add_node(author)
        if graph.has_edge(commenter, author):
            graph[commenter][author]["weight"] += 1
        else:
            graph.add_edge(commenter, author, weight=1)

    influence = _pagerank_normalised(graph)
    community = _detect_communities(graph)
    author_main_tags = {n: graph.nodes[n].get("tags", set()) for n in graph.nodes}
    return SocialGraphContext(
        graph=graph,
        influence=influence,
        community=community,
        author_main_tags=author_main_tags,
    )


def _as_user(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _pagerank_normalised(graph: nx.DiGraph) -> dict[int, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        pr = nx.pagerank(graph, weight="weight")
    except Exception:  # pragma: no cover - disconnected / numerical edge cases
        pr = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes}
    lo, hi = min(pr.values()), max(pr.values())
    if hi <= lo:
        return {n: 0.0 for n in pr}
    return {n: (v - lo) / (hi - lo) for n, v in pr.items()}


def _detect_communities(graph: nx.DiGraph) -> dict[int, int]:
    if graph.number_of_nodes() == 0:
        return {}
    undirected = graph.to_undirected()
    try:
        communities = nx.community.greedy_modularity_communities(undirected)
    except Exception:  # pragma: no cover
        communities = [set(undirected.nodes)]
    mapping: dict[int, int] = {}
    for cid, members in enumerate(communities):
        for node in members:
            mapping[node] = cid
    return mapping


# --------------------------------------------------------------------------- #
# Searcher resolution and social features
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedSearcher:
    """A concrete searcher identity grounded in the current corpus graph."""

    name: str
    intent_profile: str
    followed_authors: set[int]
    interest_tags: set[str]
    followed_communities: set[int]


def resolve_searcher(ctx: SocialGraphContext, searcher_cfg: dict, name: str = "searcher") -> ResolvedSearcher:
    """Turn a searcher CONFIG into a concrete identity in this graph.

    The searcher is connected to the most INFLUENTIAL authors whose expertise
    matches their declared interests. This models "the people in the searcher's
    social network" without hardcoding user ids, so it works on any corpus.
    """
    interest_tags = set(t.lower() for t in searcher_cfg.get("follow_tags", []))
    n_follow = int(searcher_cfg.get("n_follow", 3))

    scored: list[tuple[float, int]] = []
    for author, tags in ctx.author_main_tags.items():
        overlap = len(interest_tags & {t.lower() for t in tags})
        if overlap > 0:
            scored.append((overlap + ctx.influence.get(author, 0.0), author))
    scored.sort(reverse=True)
    followed = {author for _, author in scored[:n_follow]}
    followed_communities = {ctx.community.get(a) for a in followed if a in ctx.community}
    followed_communities.discard(None)
    return ResolvedSearcher(
        name=name,
        intent_profile=searcher_cfg.get("intent_profile", "default"),
        followed_authors=followed,
        interest_tags=interest_tags,
        followed_communities=followed_communities,
    )


def _distance_map(ctx: SocialGraphContext, sources: set[int], max_hops: int = 3) -> dict[int, int]:
    """Min hop-distance from any followed author to every reachable user."""
    undirected = ctx.undirected()
    best: dict[int, int] = {}
    for src in sources:
        if src not in undirected:
            continue
        lengths = nx.single_source_shortest_path_length(undirected, src, cutoff=max_hops)
        for node, dist in lengths.items():
            if node not in best or dist < best[node]:
                best[node] = dist
    return best


def add_searcher_features(
    results: pd.DataFrame,
    ctx: SocialGraphContext,
    searcher: ResolvedSearcher,
) -> pd.DataFrame:
    """Add social_proximity_score and community_affinity_score for a searcher."""
    out = results.copy()
    distances = _distance_map(ctx, searcher.followed_authors)

    def proximity(author: object) -> float:
        a = _as_user(author)
        if a is None:
            return 0.0
        if a in searcher.followed_authors:
            return 1.0
        dist = distances.get(a)
        if dist is None:
            return 0.0
        return math.exp(-dist)  # 1 hop ~0.37, 2 hops ~0.14, 3 hops ~0.05

    def affinity(row: pd.Series) -> float:
        author = _as_user(row.get("owner_user_id"))
        doc_tags = {t.lower() for t in tokenize(str(row.get("tags", "")))}
        tag_overlap = len(searcher.interest_tags & doc_tags) / max(1, len(searcher.interest_tags))
        community_bonus = 0.0
        if author is not None and ctx.community.get(author) in searcher.followed_communities:
            community_bonus = 0.5
        return min(1.0, tag_overlap + community_bonus)

    out["social_proximity_score"] = out.get("owner_user_id", None).map(proximity) \
        if "owner_user_id" in out else 0.0
    out["community_affinity_score"] = out.apply(affinity, axis=1)
    return out


def author_influence_series(results: pd.DataFrame, ctx: SocialGraphContext) -> pd.Series:
    """Map each candidate answer to its author's normalised influence score."""
    def lookup(author: object) -> float:
        a = _as_user(author)
        return ctx.influence.get(a, 0.0) if a is not None else 0.0

    if "owner_user_id" not in results:
        return pd.Series([0.0] * len(results), index=results.index)
    return results["owner_user_id"].map(lookup)


def influence_table(ctx: SocialGraphContext, top_n: int = 15) -> pd.DataFrame:
    """Report-ready table of the most influential authors."""
    rows = []
    for user, score in sorted(ctx.influence.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        rows.append(
            {
                "user_id": user,
                "influence_pagerank_norm": round(score, 4),
                "community": ctx.community.get(user, -1),
                "in_degree": ctx.graph.in_degree(user),
                "expertise_tags": " ".join(sorted(ctx.author_main_tags.get(user, set()))[:5]),
            }
        )
    return pd.DataFrame(rows)
