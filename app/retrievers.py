from typing import List

from langchain_core.documents import Document
from langchain_postgres import PGVector

from app.configs import PGVECTOR_URL, get_embeddings


def _make_vectorstore(collection_name: str) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=PGVECTOR_URL,
    )


customer_vectorstore = _make_vectorstore("customer_profiles")
film_vectorstore = _make_vectorstore("film_profiles")
actor_vectorstore = _make_vectorstore("actor_profiles")

customer_retriever = customer_vectorstore.as_retriever(search_kwargs={"k": 5})
film_retriever = film_vectorstore.as_retriever(search_kwargs={"k": 5})
actor_retriever = actor_vectorstore.as_retriever(search_kwargs={"k": 5})


def unified_retrieve(question: str, k_each: int = 3) -> List[Document]:
    """Query each retriever and return a combined list of docs."""

    k_each = max(1, k_each or 1)
    docs: List[Document] = []

    for retriever in (customer_retriever, film_retriever, actor_retriever):
        docs.extend(retriever.get_relevant_documents(question)[:k_each])

    return docs
