"""Data loading for Money Stack Exchange style social-search data.

Two data paths are supported:

1. REAL DATA (preferred, used for the report): the Money Stack Exchange data
   dump (https://archive.org/details/stackexchange) placed in ``data/raw/``.
   ``load_stackexchange_dump`` joins answers to their parent questions, authors
   (reputation) and comments, and crucially records the ASKER and the
   COMMENTERS of each post so the social interaction graph can be built.

2. OFFLINE SAMPLE (reproducible fallback): ``build_sample_dataset`` generates a
   structurally realistic financial Q&A corpus -- multiple authors with varied
   reputation/influence, askers, accepted answers and comment threads -- so the
   social-graph layer and the evaluation run end-to-end with no downloads. It is
   clearly labelled synthetic and is NOT the research conclusion.
"""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from . import config
from .preprocessing import clean_documents, save_processed_documents


def _read_xml_rows(path: Path) -> list[dict[str, str]]:
    """Read Stack Exchange dump XML rows without loading unrelated markup."""
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "row":
            rows.append(dict(elem.attrib))
            elem.clear()
    return rows


def _comments_index(path: Path) -> tuple[Counter, dict[int, list[int]]]:
    """Return (#comments per post, commenter user-ids per post) from Comments.xml."""
    counts: Counter = Counter()
    commenters: dict[int, list[int]] = defaultdict(list)
    for row in _read_xml_rows(path):
        post = row.get("PostId")
        if not post:
            continue
        pid = int(post)
        counts[pid] += 1
        uid = row.get("UserId")
        if uid and uid.isdigit():
            commenters[pid].append(int(uid))
    return counts, commenters


def load_stackexchange_dump(paths: config.StackExchangePaths | None = None) -> pd.DataFrame:
    """Load answers joined to their parent questions, authors and comments.

    The searchable unit is an ANSWER connected to its financial question. Social
    fields captured: community approval (Score, accepted), engagement (comments,
    views, answer count), author reputation, timestamps, plus the ASKER
    (``question_owner_user_id``) and COMMENTERS (``commenter_user_ids``) that
    define the social interaction graph.
    """
    paths = paths or config.StackExchangePaths()
    posts = _read_xml_rows(paths.posts_xml)
    if not posts:
        raise FileNotFoundError(
            f"No Stack Exchange Posts.xml found at {paths.posts_xml}. "
            "Use --sample or place the Money Stack Exchange dump in data/raw/."
        )

    users = {
        int(row["Id"]): int(row.get("Reputation", 0) or 0)
        for row in _read_xml_rows(paths.users_xml)
        if row.get("Id")
    }
    comment_counts, commenters = _comments_index(paths.comments_xml)

    questions: dict[int, dict[str, str]] = {}
    answers: list[dict[str, str]] = []
    for row in posts:
        post_type = row.get("PostTypeId")
        if post_type == "1" and row.get("Id"):
            questions[int(row["Id"])] = row
        elif post_type == "2":
            answers.append(row)

    docs: list[dict[str, object]] = []
    for answer in answers:
        question_id = int(answer.get("ParentId", 0) or 0)
        question = questions.get(question_id)
        if not question:
            continue
        answer_id = int(answer.get("Id", 0) or 0)
        owner = answer.get("OwnerUserId")
        owner_id = int(owner) if owner and owner.isdigit() else None
        asker = question.get("OwnerUserId")
        asker_id = int(asker) if asker and asker.isdigit() else None
        accepted_answer_id = question.get("AcceptedAnswerId")
        docs.append(
            {
                "doc_id": f"a{answer_id}",
                "answer_id": answer_id,
                "question_id": question_id,
                "title": question.get("Title", ""),
                "question_body": question.get("Body", ""),
                "answer_body": answer.get("Body", ""),
                "tags": question.get("Tags", ""),
                "answer_score": int(answer.get("Score", 0) or 0),
                "accepted_answer": str(answer_id) == str(accepted_answer_id),
                "comment_count": comment_counts[answer_id],
                "question_view_count": int(question.get("ViewCount", 0) or 0),
                "answer_count": int(question.get("AnswerCount", 0) or 0),
                "owner_user_id": owner_id,
                "question_owner_user_id": asker_id,
                "commenter_user_ids": " ".join(str(u) for u in commenters.get(answer_id, [])),
                "author_reputation": users.get(owner_id or -1, 0),
                "creation_date": answer.get("CreationDate", ""),
                "last_activity_date": answer.get("LastActivityDate") or question.get("LastActivityDate", ""),
                "source": "money.stackexchange.dump",
                "url": f"https://money.stackexchange.com/questions/{question_id}#answer-{answer_id}",
            }
        )
    return pd.DataFrame(docs)


# --------------------------------------------------------------------------- #
# Reproducible offline corpus (synthetic but structurally realistic)
# --------------------------------------------------------------------------- #
# Author pool: a few high-reputation "experts" who answer across many topics
# (these become graph hubs), plus topic specialists. (user_id, reputation,
# primary_aspect, is_generalist)
_AUTHORS = [
    (301, 41000, "investing", True),
    (302, 38000, "tax", True),
    (303, 35000, "risk", True),
    (304, 22000, "investing", False),
    (305, 19000, "saving", False),
    (306, 17000, "mortgage", False),
    (307, 15000, "retirement", False),
    (308, 12000, "credit", False),
    (309, 9000, "debt", False),
    (310, 7000, "saving", False),
    (311, 5000, "tax", False),
    (312, 3400, "debt", False),
    (313, 2100, "credit", False),
    (314, 900, "mortgage", False),
]

# (title, tags, aspect, [answer snippets, best-first])
_QUESTION_BANK = [
    ("Should I pay off credit card debt before investing?", "credit-card debt investing", "debt",
     ["Pay off high-interest credit card debt before investing. A guaranteed ~20% avoided interest usually beats uncertain market returns.",
      "Contribute enough to capture any employer match, then prioritise the card balance and a small emergency buffer.",
      "Compare the card APR with your expected after-tax return. Above roughly 8-10% APR, paying the debt almost always wins."]),
    ("How do I get out of credit card debt fastest?", "credit-card debt budgeting", "debt",
     ["List balances, attack the highest APR first (avalanche), and stop new charges while you repay.",
      "The snowball method (smallest balance first) can help motivation even if it costs slightly more interest."]),
    ("How much emergency fund should I keep?", "emergency-fund saving budgeting", "saving",
     ["Three to six months of essential expenses, kept liquid in a high-yield savings account, not in volatile assets.",
      "If your income is irregular or your job is unstable, target the higher end of the range."]),
    ("Where should I keep my emergency savings?", "saving emergency-fund risk", "saving",
     ["Safety and liquidity first: high-yield savings or short Treasury bills, not equities you may have to sell at a loss.",
      "Split it: one month in checking for instant access, the rest in a high-yield account."]),
    ("Is it better to rent or buy a house?", "mortgage rent buy housing", "mortgage",
     ["Compare unrecoverable costs: rent vs mortgage interest, taxes, insurance, maintenance and transaction costs. Buying is not automatically better.",
      "Use a rent-vs-buy calculator and stress-test it against moving within five years."]),
    ("Should I refinance my mortgage?", "mortgage refinance interest-rate", "mortgage",
     ["Refinance when interest savings exceed closing costs and you will keep the loan past the break-even point.",
      "Watch the term reset: lowering the rate but restarting 30 years can raise total interest paid."]),
    ("How are capital gains taxed?", "tax capital-gains investing", "tax",
     ["It depends on holding period, income, jurisdiction and cost basis. Separate short-term from long-term and keep records.",
      "Losses can offset gains, but wash-sale rules and local law change the result; consult a professional for large trades."]),
    ("What records do I need for investment taxes?", "tax investing capital-gains", "tax",
     ["Keep trade confirmations, cost basis, dividend statements and broker 1099s; reconstruct basis before you sell.",
      "Most brokers track basis now, but verify it for transferred or inherited positions."]),
    ("Should I invest in ETFs or individual stocks?", "etf stocks investing diversification", "investing",
     ["For most investors, diversified low-cost ETFs cut idiosyncratic risk and need less monitoring than single stocks.",
      "Individual stocks can fit if you understand concentration risk, position sizing and the tax consequences.",
      "A core ETF portfolio plus a small satellite of stocks is a common compromise."]),
    ("Is dollar cost averaging better than lump sum?", "investing dollar-cost-averaging risk", "investing",
     ["Historically lump sum has a higher expected return, but averaging reduces timing risk and regret.",
      "If the money is already in cash, averaging it in over a few months is a reasonable behavioural compromise."]),
    ("What fees matter when choosing an ETF?", "etf investing fees", "investing",
     ["Expense ratio first, then bid-ask spread, tracking error and any platform commission.",
      "A 0.5% vs 0.05% expense ratio compounds into a large gap over decades."]),
    ("What should I do if I think an investment is a scam?", "scam fraud risk investing", "risk",
     ["Stop sending money, preserve all records, verify registration with the regulator, and report the suspected fraud.",
      "Guaranteed high returns and pressure to act fast are classic warning signs; a cautious answer beats optimism here."]),
    ("How do I avoid investment fraud?", "scam fraud risk", "risk",
     ["Check the seller's registration, be sceptical of guaranteed returns, and never let urgency rush a decision.",
      "If you cannot independently verify where returns come from, treat it as a red flag."]),
    ("Should I use a Roth or traditional retirement account?", "retirement tax roth ira", "retirement",
     ["Compare your marginal tax rate now with the rate you expect in retirement; Roth favours paying tax while rates are low.",
      "Many people split contributions to hedge against future tax-rate uncertainty."]),
    ("When should I prioritise retirement contributions?", "retirement investing saving", "retirement",
     ["After high-interest debt and a starter emergency fund, capture the full employer match before anything else.",
      "Tax-advantaged space is use-it-or-lose-it each year, so fill it early when you can."]),
    ("How can I improve my credit score?", "credit-score credit-card debt", "credit",
     ["Pay on time, keep utilisation low, avoid unnecessary applications and check your reports for errors.",
      "Length of history helps, so keep your oldest card open even if you rarely use it."]),
    ("How much credit utilisation is healthy?", "credit-score credit-card", "credit",
     ["Keeping reported utilisation under about 30%, and ideally under 10%, helps your score.",
      "Paying before the statement date lowers the balance that gets reported."]),
    ("Should I keep cash or buy bonds for short-term savings?", "saving bonds treasury emergency-fund", "saving",
     ["For short horizons, liquidity and safety beat yield; Treasury bills or a high-yield account usually fit.",
      "Match the bond maturity to when you need the money to avoid selling at a loss."]),
    ("Do I need a financial advisor?", "advisor investing retirement fees", "investing",
     ["A fee-only fiduciary helps with complex tax, estate or retirement planning; simple investing may only need a low-cost index fund.",
      "Avoid commission-based salespeople; understand exactly how any advisor is paid."]),
    ("How do I start investing with little money?", "investing etf beginner", "investing",
     ["Start with a broad low-cost index ETF and automatic monthly contributions; consistency matters more than size.",
      "Many brokers allow fractional shares, so you can begin with very small amounts."]),
    ("Should I pay extra on my mortgage or invest?", "mortgage investing debt", "mortgage",
     ["Compare the mortgage rate with your expected after-tax return and your risk tolerance; there is no single right answer.",
      "Paying down a low fixed-rate mortgage is a guaranteed but modest return; investing has higher expected but uncertain returns."]),
    ("How do I budget on an irregular income?", "budgeting saving emergency-fund", "saving",
     ["Budget on your lowest typical month, hold a larger buffer, and pay yourself a steady 'salary' from a holding account.",
      "Separate fixed bills from variable spending so lean months are predictable."]),
]


def build_sample_dataset(seed: int = 7) -> pd.DataFrame:
    """Create a structurally realistic financial Q&A sample that runs offline."""
    rng = random.Random(seed)
    authors_by_aspect: dict[str, list[int]] = defaultdict(list)
    generalists = [a[0] for a in _AUTHORS if a[3]]
    rep_by_author = {a[0]: a[1] for a in _AUTHORS}
    for uid, _rep, aspect, _gen in _AUTHORS:
        authors_by_aspect[aspect].append(uid)

    askers = list(range(401, 441))
    docs = []
    answer_id = 1000
    for qi, (title, tags, aspect, snippets) in enumerate(_QUESTION_BANK, start=1):
        question_id = 500 + qi
        asker_id = askers[qi % len(askers)]
        # candidate answerers: topic specialists for this aspect + some generalists
        pool = authors_by_aspect.get(aspect, []) + generalists
        rng.shuffle(pool)
        n_answers = len(snippets)
        chosen = pool[:n_answers]
        # the accepted answer goes to the highest-reputation chosen author
        accepted_author = max(chosen, key=lambda u: rep_by_author.get(u, 0))
        base_views = rng.randint(8000, 52000)
        for ai, snippet in enumerate(snippets):
            answer_id += 1
            author = chosen[ai % len(chosen)]
            is_accepted = author == accepted_author and ai == chosen.index(accepted_author)
            score = rng.randint(12, 30) + (20 if is_accepted else 0) + int(rep_by_author[author] / 4000)
            comment_count = rng.randint(0, 9)
            # commenters: the asker plus a few other authors -> drives the graph
            commenters = [asker_id] + rng.sample(generalists, k=min(2, len(generalists)))
            commenters = [c for c in commenters if c != author][: max(0, min(comment_count, 3))]
            year = rng.choice([2022, 2023, 2024, 2024, 2025, 2025])
            month = rng.randint(1, 12)
            date = f"{year}-{month:02d}-15"
            docs.append(
                {
                    "doc_id": f"sample-a{answer_id}",
                    "answer_id": answer_id,
                    "question_id": question_id,
                    "title": title,
                    "question_body": f"<p>Community question about {title.lower()}</p>",
                    "answer_body": f"<p>{snippet}</p>",
                    "tags": tags,
                    "answer_score": score,
                    "accepted_answer": is_accepted,
                    "comment_count": comment_count,
                    "question_view_count": base_views,
                    "answer_count": n_answers,
                    "owner_user_id": author,
                    "question_owner_user_id": asker_id,
                    "commenter_user_ids": " ".join(str(c) for c in commenters),
                    "author_reputation": rep_by_author[author],
                    "creation_date": date,
                    "last_activity_date": date,
                    "source": "offline.sample",
                    "url": f"https://money.stackexchange.com/questions/{question_id}#answer-{answer_id}",
                    "aspect": aspect,
                }
            )
    return pd.DataFrame(docs)


def build_or_load_documents(use_sample: bool = False) -> pd.DataFrame:
    """Load real dump files when available, otherwise use the sample dataset."""
    config.ensure_directories()
    if use_sample or not config.StackExchangePaths().posts_xml.exists():
        print("Using reproducible offline financial social-search sample dataset.")
        raw = build_sample_dataset()
    else:
        print("Loading Money Stack Exchange dump from data/raw/.")
        raw = load_stackexchange_dump()
    docs = clean_documents(raw)
    save_processed_documents(docs)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare financial social-search documents.")
    parser.add_argument("--sample", action="store_true", help="force the bundled sample dataset")
    args = parser.parse_args()
    docs = build_or_load_documents(use_sample=args.sample)
    print(f"Prepared {len(docs)} answer documents -> {config.PROCESSED_CSV}")


if __name__ == "__main__":
    main()
