# 15-Minute Presentation Outline (Social Search)

The professor requires ~15 minutes and that **each group member presents an
aspect**. Below is a 3-member split (merge sections for a 2-person group).

## Member A — Objective & Data (≈5 min)
- The social-search problem and domain: personal-finance community Q&A; the
  retrievable unit is an **answer** plus its social entities (asker, answerers,
  commenters). *(slides: What is Social Search / Retrievable information)*
- Why this data is *suitable for social search*: votes, accepted answers,
  reputation, comments, asker/commenter ids → social relationships + influence.
- Real Money Stack Exchange loader vs the reproducible offline corpus.
- Figures: `pipeline_diagram.png`.

## Member B — Models (≈5 min)
- Topical baselines: BM25 and TF-IDF (the candidate generators).
- The transparent social re-ranker: the weighted feature fusion and why topical
  keeps the largest weight.
- **The social graph**: interaction graph, **PageRank influence**, community
  detection; searcher personalization (social proximity + community affinity) →
  *same query, different searchers → different results*.
- Figures: `user_interaction_or_author_influence.png`,
  `social_feature_contribution.png`, `searcher_personalization_example.png`,
  `before_after_ranking_example.png`.

## Member C — Evaluation & Discussion (≈5 min)
- Evaluation framework: TREC-style pooling, **model-independent dual gold**
  (topical + social usefulness + combined), and the anti-circularity argument.
- Metrics: P@k, Recall@10, MAP, nDCG@k for BM25 / TF-IDF / BM25+Social.
- Headline result: +0.257 nDCG@10 on social-usefulness gold, small topicality
  cost; ablation shows the social graph contributes.
- Limitations & future work; the human-annotation template for the final gold.
- Figures: `evaluation_metrics.png`, `ablation_chart.png`.

## Closing (≈ shared)
- Mapping to the rubric (`report_assets/tables/grading_requirements.csv`) and
  honest statement of what is synthetic vs real.
