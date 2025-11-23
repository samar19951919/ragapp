# app/main.py

from typing import List, Optional, Literal

from typing import Dict
from langchain_core.documents import Document
from app.retrievers import unified_retrieve
from app.helpers import call_llm_with_docs, docs_summary_for_response


from fastapi import FastAPI
from pydantic import BaseModel

from app.helpers import call_llm_with_docs, docs_summary_for_response
from app.retrievers import (
    customer_retriever,
    film_retriever,
    actor_retriever,
    unified_retrieve,
)


app = FastAPI(
    title="RAG Sakila API",
    description="RAG over customers, films, and actors using PGVector + LangChain.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class AskUnifiedRequest(BaseModel):
    question: str
    top_k_each: Optional[int] = 3
    scope: Optional[List[Literal["customers", "films", "actors"]]] = None  # not used yet, but ready


class AskResponse(BaseModel):
    answer: str
    used_docs: List[dict]


@app.post("/ask/customers", response_model=AskResponse)
def ask_customers(payload: AskRequest):
    docs = customer_retriever.get_relevant_documents(payload.question)
    docs = docs[: payload.top_k]
    answer = call_llm_with_docs(payload.question, docs)
    used_docs = docs_summary_for_response(docs)
    return AskResponse(answer=answer, used_docs=used_docs)


@app.post("/ask/films", response_model=AskResponse)
def ask_films(payload: AskRequest):
    docs = film_retriever.get_relevant_documents(payload.question)
    docs = docs[: payload.top_k]
    answer = call_llm_with_docs(payload.question, docs)
    used_docs = docs_summary_for_response(docs)
    return AskResponse(answer=answer, used_docs=used_docs)


@app.post("/ask/actors", response_model=AskResponse)
def ask_actors(payload: AskRequest):
    docs = actor_retriever.get_relevant_documents(payload.question)
    docs = docs[: payload.top_k]
    answer = call_llm_with_docs(payload.question, docs)
    used_docs = docs_summary_for_response(docs)
    return AskResponse(answer=answer, used_docs=used_docs)


@app.post("/ask/unified", response_model=AskResponse)
def ask_unified(payload: AskUnifiedRequest):
    docs = unified_retrieve(payload.question, k_each=payload.top_k_each or 3)
    answer = call_llm_with_docs(payload.question, docs)
    used_docs = docs_summary_for_response(docs)
    return AskResponse(answer=answer, used_docs=used_docs)


@app.get("/health")
def health():
    return {"status": "ok"}

# app/main.py (add models)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    session_id: str        # client-generated ID for the chat (e.g. uuid)
    message: str           # new user message
    top_k_each: int = 3


class ChatResponse(BaseModel):
    answer: str
    used_docs: List[dict]
    session_id: str

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """
    Multi-turn chat over your RAG system.
    session_id groups messages into one conversation.
    """

    # 1) Get or init history
    history = chat_histories.get(payload.session_id, [])

    # 2) Append new user message
    history.append(ChatMessage(role="user", content=payload.message))

    # 3) Use ONLY the latest user message for retrieval
    #    (you can get fancy later and include more context)
    last_user_msg = payload.message
    docs = unified_retrieve(last_user_msg, k_each=payload.top_k_each)

    # 4) Build a richer prompt with chat history + docs
    #    Here we just prepend history to the question string
    history_text = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in history[-6:]  # last few turns
    )
    question_with_history = (
        f"Conversation so far:\n{history_text}\n\nCurrent user message: {last_user_msg}"
    )

    answer = call_llm_with_docs(question_with_history, docs)
    used_docs = docs_summary_for_response(docs)

    # 5) Save assistant reply in history
    history.append(ChatMessage(role="assistant", content=answer))
    chat_histories[payload.session_id] = history

    return ChatResponse(
        answer=answer,
        used_docs=used_docs,
        session_id=payload.session_id,
    )
