# app/helpers.py

from typing import List

from langchain_core.documents import Document

from app.configs import get_pg_connection, get_llm


# ---- DB helpers ----

def fetch_view_rows(view_name: str):
    """
    Fetch all rows from a given view and return (rows, column_names).
    """
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {view_name};")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return rows, cols


# ---- RAG prompt helpers ----

def build_context_from_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        snippet = d.page_content
        block = f"[DOC {i}] METADATA: {meta}\nCONTENT:\n{snippet}\n"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def call_llm_with_docs(question: str, docs: List[Document]) -> str:
    llm = get_llm()
    context = build_context_from_docs(docs)

    prompt = f"""
You are a helpful assistant for a DVD rental company.

User question:
{question}

Here is relevant context from the knowledge base (customers, films, actors):
{context}

Use only this information to answer the question.
If the answer is not clearly contained here, say you cannot answer based on the available data.
Respond clearly and concisely.
"""

    resp = llm.invoke(prompt)
    return resp.content


def docs_summary_for_response(docs: List[Document]):
    """
    Convert docs to JSON-friendly summaries for API response/debugging.
    """
    out = []
    for d in docs:
        out.append(
            {
                "metadata": d.metadata,
                "snippet": d.page_content[:300] + ("..." if len(d.page_content) > 300 else ""),
            }
        )
    return out
