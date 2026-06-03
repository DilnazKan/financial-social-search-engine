# Grading Checklist (Social Search, max 8 points)

| Rubric item (max) | Status | Evidence |
|---|---|---|
| **Objective** — completeness & clarity (1) | ✅ | README §1; domain + goal in professor's terms |
| **Data** — correctness (1) | ✅ | Real SE-dump loader with asker/commenter/votes/reputation + reproducible offline corpus (`src/data_collection.py`) |
| **Models** — correctness (2.5) | ✅ | BM25 + TF-IDF; transparent social re-ranker; **social graph** (PageRank, communities) + searcher personalization (`src/social_graph.py`, `src/reranking.py`) |
| **Evaluation** — metrics (2.5) | ✅ | P@k, R@k, MAP, nDCG@k; model-independent **dual gold**; TREC pooling; ablation (`src/evaluation.py`) |
| **Evaluation** — discussion (1) | ✅ | `results/metrics/evaluation_discussion.md`; anti-circularity note; human template |

## Submission (slides p.16)
- [ ] Final report discussing Objective / Data / Models / Evaluation
- [ ] Source code + README (this repo)
- [ ] Data (processed + sample; real dump instructions in README §3)
- [ ] ~15-min PowerPoint, **each member presents an aspect** (outline provided)
- [ ] Google Drive folder named `WSA_surname1_surname2_...`, shared with the
      addresses on slide p.16
