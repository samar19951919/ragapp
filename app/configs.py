# app/configs.py

import os
from functools import lru_cache

import psycopg2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ---- ENV + URLS ----

# Ensure values from a local .env file are loaded when running outside a fully
# configured cloud environment (e.g., local development or App Runner preview
# tasks). This is a no-op if the file does not exist.
load_dotenv()

PGVECTOR_URL = os.getenv("PG_CONNECTION_STRING")
if not PGVECTOR_URL:
    raise ValueError("PG_CONNECTION_STRING env var is not set")



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env var is not set")


# ---- DB CONNECTION ----

@lru_cache(maxsize=1)
def get_pg_connection():
    """
    Single shared psycopg2 connection.
    In a real prod setup you might use a pool instead.
    """
    conn = psycopg2.connect(PGVECTOR_URL)
    conn.autocommit = True
    return conn


# ---- LangChain shared objects ----

@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()  # uses OPENAI_API_KEY from env


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    # tune model as you like
    return ChatOpenAI(model="gpt-4o")
