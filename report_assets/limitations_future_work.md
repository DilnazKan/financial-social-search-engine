# Limitations and Future Work

## Limitations
- **Offline corpus is synthetic.** The bundled 46-answer corpus is deterministic
  and structurally realistic but generated; quantitative numbers should be
  reproduced on the real Money Stack Exchange dump before being reported as
  findings. The loader for the real dump is provided and tested on its schema.
- **Reference labels are automatic.** The dual-gold labels are model-INDEPENDENT
  but heuristic. Community signals appear both as a ranking feature and inside
  the social-usefulness gold, which advantages the social run on that gold; this
  is mitigated by also reporting the independent topical gold and by shipping a
  human-annotation template for the definitive evaluation.
- **Graph from one platform.** Influence/communities come from a single Q&A
  site's asker/answerer/commenter interactions; cross-platform follows are not
  modelled.
- **Linear fusion.** Weights are hand-set and interpretable rather than learned.

## Future work
- Run on the full real dump and complete a human-judged qrels (the template is
  ready) for headline numbers.
- Learning-to-rank (e.g. LambdaMART) to learn feature weights, compared against
  the transparent linear baseline.
- Richer graph signals: HITS hub/authority, temporal influence, true follow
  edges where available.
- Per-searcher evaluation with searcher-specific judgments to quantify
  personalization, not only illustrate it.
- Dense / hybrid retrieval (BM25 + embeddings) as a stronger topical stage.
