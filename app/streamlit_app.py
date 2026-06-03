"""Optional Streamlit app for Financial Social Search."""

import streamlit as st

from src.preprocessing import load_processed_documents
from src.retrieval import retrieve_candidates
from src.reranking import apply_social_reranking


st.set_page_config(page_title="Financial Social Search", layout="wide")
st.title("Financial Social Search Engine")
st.caption("BM25 candidates reranked by community, engagement, credibility, freshness, tags and sentiment alignment.")

query = st.text_input("Query", "Should I pay credit card debt before investing?")
profile = st.selectbox("Profile", ["default", "beginner", "advanced_investor", "risk_sensitive"])
method = st.selectbox("Baseline", ["bm25", "tfidf"])
top_k = st.slider("Top K", 5, 20, 10)

docs = load_processed_documents()
if query:
    candidates = retrieve_candidates(docs, query, method, 100)
    results = apply_social_reranking(candidates, query, profile=profile).head(top_k)
    st.dataframe(results, use_container_width=True)

