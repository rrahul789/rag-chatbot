"""Streamlit chat UI. Talks to the FastAPI backend over HTTP, so it can be
deployed/scaled independently and swapped for a different frontend later.

Run: streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="centered")
st.title("📚 Document Q&A Chatbot")
st.caption("Ask questions about the ingested document collection. Answers are grounded in retrieved context only.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

with st.sidebar:
    st.header("Knowledge base")
    st.write("Upload documents to add them to the vector store, or use the bundled sample docs.")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / MD / DOCX", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
    )
    if st.button("Ingest uploaded files", disabled=not uploaded_files):
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        with st.spinner("Chunking, embedding and storing..."):
            resp = requests.post(f"{API_URL}/ingest", files=files_payload, timeout=120)
        if resp.ok:
            data = resp.json()
            st.success(f"Ingested {data['chunks_ingested']} chunks from {len(data['files_processed'])} file(s).")
        else:
            st.error(f"Ingest failed: {resp.text}")

    if st.button("Load bundled sample docs"):
        with st.spinner("Ingesting sample_docs/..."):
            resp = requests.post(f"{API_URL}/ingest/sample-docs", timeout=120)
        if resp.ok:
            data = resp.json()
            st.success(f"Ingested {data['chunks_ingested']} chunks from {data['files_processed']}.")
        else:
            st.error(f"Ingest failed: {resp.text}")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for src in message["sources"]:
                    page = f", page {src['page']}" if src.get("page") is not None else ""
                    st.markdown(f"**{src['source']}**{page} — relevance {src['score']}")
                    st.caption(src["snippet"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history_payload = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "chat_history": history_payload},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander(f"Sources ({len(data['sources'])})"):
                        for src in data["sources"]:
                            page = f", page {src['page']}" if src.get("page") is not None else ""
                            st.markdown(f"**{src['source']}**{page} — relevance {src['score']}")
                            st.caption(src["snippet"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "sources": data.get("sources", [])}
                )
            except requests.exceptions.RequestException as exc:
                error_text = f"Could not reach the API at {API_URL}: {exc}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
