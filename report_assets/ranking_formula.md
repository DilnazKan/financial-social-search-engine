# Ranking Formula

## Stage 1 — Topical candidate retrieval
BM25 (or TF-IDF + cosine) over tokenized `combined_text` produces the candidate
set and `topical_score ∈ [0,1]`.

## Stage 2 — Social re-ranking (linear fusion)
For each candidate answer *d*:

```
final(d) = ( Σ_f  w_f · feature_f(d) ) / Σ_f w_f
```

All features are scaled to `[0,1]`. Default weights (`config.DEFAULT_SOCIAL_WEIGHTS`):

| feature | weight | type |
|---|---|---|
| topical_score | 0.50 | topical anchor |
| community_score | 0.14 | social (approval) |
| credibility_score | 0.08 | social (author authority) |
| author_influence_score | 0.07 | **social graph (PageRank)** |
| engagement_score | 0.06 | social (attention) |
| freshness_score | 0.05 | temporal |
| tag_match_score | 0.05 | social (community topics) |
| sentiment_alignment_score | 0.05 | stance usefulness |

## Stage 3 — Searcher personalization (when `--searcher` is set)
Two social-graph signals are added and the searcher's intent profile is used:

| feature | weight | meaning |
|---|---|---|
| social_proximity_score | 0.13 | closeness to the searcher's followed authors in the graph |
| community_affinity_score | 0.07 | overlap with the searcher's interest communities |

Intent profiles (`beginner`, `advanced_investor`, `risk_sensitive`) re-weight
the same signals for different information needs.
