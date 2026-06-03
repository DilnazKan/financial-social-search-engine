# Evaluation discussion (auto-generated)

Corpus: 46 answer documents. Queries: 18. Social graph: 33 users, 62 interaction edges, 6 detected communities.

## Headline numbers (nDCG@10)

- Social-usefulness gold: BM25 = 0.605 vs BM25+Social = 0.862 (delta +0.257).
- Topical gold (fairness check): BM25 = 0.884 vs BM25+Social = 0.834 (delta -0.051).
- Combined gold: BM25 = 0.839 vs BM25+Social = 0.851 (delta +0.012).

## How to read this

The social re-ranker is expected to help most on the SOCIAL-usefulness gold, because it incorporates community approval, author credibility and graph influence: it promotes community-validated answers that a purely lexical ranker places lower. On the TOPICAL gold the two systems should be close, confirming that social re-ranking reorders within topically relevant candidates rather than sacrificing topicality.

## Honesty / anti-circularity note

Relevance labels are computed once, from query/document content (topical) and real community signals (social), and never depend on any system's own ranks or scores. Because community signals appear both as ranking features and inside the social gold, the social re-ranker has a built-in advantage on that gold; this is why we also report the independent topical gold and provide an empty human-annotation template (`data/qrels/social_judgments_template.csv`) for the definitive report evaluation.

## Ablation (combined gold, nDCG@10)

- full_social: 0.851
- without_community_score: 0.861
- without_credibility_score: 0.846
- without_author_influence_score: 0.838
- without_freshness_score: 0.848
- without_engagement_score: 0.847
- without_sentiment_alignment_score: 0.849