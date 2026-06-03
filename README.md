# Financial Social Search Engine

A **social search engine** for personal-finance question answering. Given a
natural-language financial question, it retrieves community answers and ranks
them not only by topical relevance but by **social evidence** — community
approval, author credibility, **social-graph influence**, engagement, freshness,
topic communities — and, for a known searcher, by the searcher's **social
relationships** to the content authors.

The design follows Prof. Viviani's definition of social search (course slides,
section *Social Search*): *"social search is a personalized search technology
with online-community filtering … social search systems also consider the social
relationships between the searcher and the results … and influence metrics of
authors."* The engine implements exactly these elements.

---

## 1. Objective 

**Domain.** Personal-finance / investing community Q&A (Money Stack Exchange
style). The retrievable unit is an **answer** connected to its question, author,
askers and commenters — i.e. user-generated content on a social platform, with
the surrounding social entities (slides *Retrievable information*).

**Goal.** Return the answers that are both *on-topic* and *socially useful*:
endorsed by the community, written by credible/influential members, and — when
the searcher is known — close to the searcher in the social graph. This moves
the engine from "one size fits all" topical search to personalized,
community-filtered social search.

---

## 2. Why this is social search 

| Сoncept  | Where it lives in this project |
|---|---|
| Social relationships between searcher and results (follows, shared interests) | `social_proximity_score`, `community_affinity_score` in `src/social_graph.py` |
| Online-community filtering / personalized results | searcher-aware re-ranking in `src/reranking.py` (`--searcher`) |
| Influence metrics of authors | PageRank over the user interaction graph → `author_influence_score` |
| Social-graph indexing (graph DB / adjacency) | `networkx` interaction graph in `src/social_graph.py` |
| Engagement-metric indexing (comments, views) | `engagement_score` in `src/social_features.py` |
| Community detection / shared-interest groups | greedy-modularity communities in `src/social_graph.py` |
| Textual indexing (tokenize, normalize, stopwords) | `src/preprocessing.py`, BM25/TF-IDF in `src/retrieval.py` |
| Temporal metadata | `freshness_score` |

---

## 3. Data 

**Primary (real) source.** The Money Stack Exchange data dump
(`https://archive.org/details/stackexchange`). Place `Posts.xml`, `Users.xml`,
`Comments.xml` (and optionally `Votes.xml`) in `data/raw/` and the loader
(`src/data_collection.py::load_stackexchange_dump`) joins each **answer** to:

* its **question** (title, body, tags, view count, answer count);
* **community approval**: answer `Score`, accepted-answer flag;
* **author**: `OwnerUserId` and reputation (from `Users.xml`);
* **social graph inputs**: the **asker** (`question_owner_user_id`) and the
  **commenters** (`commenter_user_ids`, from `Comments.xml`), which define the
  user interaction edges.

This is what makes the data *suitable for social search*: it carries the social
relationships and influence signals, not just text.

**Reproducible offline corpus.** Because the full dump is large and not always
available, `build_sample_dataset()` generates a **structurally realistic**
finance Q&A corpus (46 answers, ~22 questions, 14 authors with varied
reputation, askers and comment threads) so the whole pipeline — including the
social graph and evaluation — runs with **no downloads**. It is deterministic
(`seed=7`) and clearly labelled synthetic; it is the runnable demonstration, not
the research claim.

```bash
python -m src.data_collection            # uses real dump if data/raw/ present
python -m src.data_collection --sample   # force the offline corpus
```

---

## 4. Models 

### 4.1 Topical retrieval (baselines)
* **BM25** (`rank_bm25`) and **TF-IDF + cosine** (`scikit-learn`) over the
  tokenized `combined_text` (title + question + answer + tags). These produce the
  candidate set and the `topical_score`.

### 4.2 Social re-ranker (post-processing personalization)
A transparent **linear fusion** re-orders the topical candidates
(`src/reranking.py`). Every signal is in `[0,1]`; topical relevance always keeps
the largest weight so social evidence re-ranks *relevant* answers rather than
replacing relevance:

```
final(d) = Σ_f  w_f · feature_f(d)        (weights normalized to sum 1)
```

Default social features (`src/social_features.py`, `src/config.py`):

| feature | computed from | social meaning |
|---|---|---|
| `topical_score` | BM25 / TF-IDF | on-topic anchor |
| `community_score` | answer votes + accepted | community approval |
| `credibility_score` | log author reputation | author authority |
| `author_influence_score` | **PageRank over the interaction graph** | network influence |
| `engagement_score` | comments, views, answer count | attention / discussion |
| `freshness_score` | exp(-age / half-life) | temporal relevance |
| `tag_match_score` | query↔tag Jaccard | community topic match |
| `sentiment_alignment_score` | intent + financial stance | usefulness of stance (cautious answers for risk/scam) |

### 4.3 Social graph + searcher personalization (the core social layer)
`src/social_graph.py` builds a **directed user interaction graph**: `asker →
answerer` and `commenter → author` edges. From it we derive:

* **author influence** — normalized PageRank (`author_influence_score`);
* **communities** — greedy-modularity clusters (shared-interest groups);
* a **searcher** identity (`src/config.py::SEARCHER_PROFILES`) resolved at
  runtime to the most influential authors matching the searcher's interests
  (no hardcoded user ids). For a searcher, two extra signals fire:
  * `social_proximity_score` — graph closeness to the searcher's followed
    authors (1.0 if followed, decaying by hop distance);
  * `community_affinity_score` — overlap with the searcher's interest communities.

This delivers the slide's defining property: **the same query is ranked
differently for different searchers** based on their social network.

```bash
python -m src.demo_search --query "How should I start investing?" --searcher active_investor
python -m src.demo_search --query "How should I start investing?" --searcher cautious_planner
```

---

## 5. Evaluation 

The evaluation (`src/evaluation.py`) is **model-independent** and avoids the
classic circularity trap (labels must never come from the system being judged).

### 5.1 Relevance judgments (model-independent, dual gold)
For each of 18 queries, the judged pool is the **union of every system's top-k**
(TREC-style pooling). Each (query, answer) gets graded labels (0–3) computed
once, from content and real community signals — never from any system's ranks:

* **`topical_gold`** — independent topical relevance (query ↔ title+tags overlap
  and aspect agreement). *Fairness check.*
* **`social_gold`** — community-endorsed usefulness (accepted + votes +
  reputation, normalized within the pool). *The social contribution.*
* **`combined_gold`** — holistic annotator (useful ⇒ must be on-topic).

A blank **human-annotation template** (`data/qrels/social_judgments_template.csv`)
is provided for the definitive report evaluation; the reference labels above
illustrate the rubric.

### 5.2 Metrics
Precision@{5,10}, Recall@10, **MAP**, **nDCG@{5,10}** (graded gains), for
**BM25 vs TF-IDF vs BM25+Social**, against **each** gold.

### 5.3 Results (offline corpus, 46 docs, 18 queries)
nDCG@10:

| gold | BM25 | TF-IDF | **BM25 + Social** |
|---|---|---|---|
| social-usefulness | 0.605 | 0.593 | **0.862** |
| topical (fairness) | 0.884 | 0.931 | 0.834 |
| combined | 0.839 | 0.878 | 0.851 |

**Reading:** social re-ranking gives a large gain on social usefulness
(**+0.257 nDCG@10, MAP 0.72 vs 0.40**) — it surfaces community-validated answers
that lexical search buries — at a small, expected topicality cost (−0.05), and a
slight net gain on the combined gold. This is the honest social-IR trade-off.

### 5.4 Ablation (combined gold)
Removing each social feature one at a time; removing **author_influence** lowers
nDCG@10 (0.851 → 0.838), evidence the **social graph contributes**.

### 5.5 Anti-circularity statement
Labels never depend on any ranker's output. Because community signals appear
both as a ranking feature and inside the social gold, we explicitly also report
the independent topical gold and ship the human-annotation template. This is
discussed in `results/metrics/evaluation_discussion.md` (auto-generated).

---

## 6. How to run

```bash
pip install -r requirements.txt

python -m src.data_collection --sample        # 1. prepare corpus
python -m src.demo_search --query "How are capital gains taxed?"   # 2. search
python -m src.demo_search --query "How should I start investing?" --searcher active_investor  # 2b. personalized
python -m src.evaluation                       # 3. dual-gold metrics + ablation
python -m src.visualization                    # 4. figures + report tables
```

Outputs: `results/metrics/*.csv`, `results/figures/*.png`,
`report_assets/tables/*.csv`. The pipeline runs end-to-end on the offline corpus
with no network access.

---

## 7. Repository structure

```
src/
  config.py            paths, weights, searcher profiles
  data_collection.py   real SE-dump loader + reproducible offline corpus
  preprocessing.py     cleaning, tokenization, combined_text
  retrieval.py         BM25 + TF-IDF baselines
  social_features.py   per-document social features (+ graph influence)
  social_graph.py      interaction graph, PageRank, communities, searcher proximity
  reranking.py         transparent weighted fusion (+ searcher personalization)
  evaluation.py        model-independent dual-gold metrics + ablation
  visualization.py     figures + report tables
  demo_search.py       CLI demo (--searcher enables social-graph personalization)
data/        raw/ (real dump), processed/, sample/, qrels/
results/     runs/, metrics/, figures/
report_assets/ figures/, tables/, *.md
```

## 8. Deliverable checklist
Report (Objective/Data/Models/Evaluation) · this code + README · data ·
~15-min PowerPoint where **each member presents an aspect** — see
`report_assets/presentation_outline_15min.md`.
