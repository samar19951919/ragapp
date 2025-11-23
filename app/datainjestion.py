# app/dataingestion.py

from app.configs import PGVECTOR_URL, get_embeddings
from app.helpers import fetch_view_rows
from langchain_postgres import PGVector


# ---- Vector stores for ingestion ----

embeddings = get_embeddings()

customer_vs = PGVector(
    embeddings=embeddings,
    collection_name="customer_profiles",
    connection=PGVECTOR_URL,
)

film_vs = PGVector(
    embeddings=embeddings,
    collection_name="film_profiles",
    connection=PGVECTOR_URL,
)

actor_vs = PGVector(
    embeddings=embeddings,
    collection_name="actor_profiles",
    connection=PGVECTOR_URL,
)


# ---- CUSTOMER INGESTION ----

def build_customer_docs():
    rows, cols = fetch_view_rows("customer_profile_docs_with_id")

    texts = []
    metadatas = []

    for row in rows:
        rec = dict(zip(cols, row))

        full_name = f"{rec['first_name']} {rec['last_name']}"
        address_parts = [rec["address"], rec.get("address2"), rec["district"], rec["city"], rec["country"]]
        address_str = ", ".join([p for p in address_parts if p])

        text = (
            f"Customer profile for {full_name} "
            f"(customer_id={rec['customer_id']}, email={rec['email']}). "
            f"They are associated with store {rec['store_id']} and live at {address_str}. "
            f"Their phone number is {rec['phone'] or 'N/A'}. "
            f"They have made {rec['total_rentals']} rentals and {rec['total_payments']} payments, "
            f"spending a total of ${rec['total_spent']:.2f}. "
            f"Their first rental was on {rec['first_rental_date']} and the most recent on {rec['last_rental_date']}. "
            f"Account active flag is {rec['active']} (created at {rec['create_date']})."
        )

        texts.append(text)
        metadatas.append(
            {
                "doc_id": str(rec["doc_id"]),
                "entity_type": "customer_profile",
                "customer_id": rec["customer_id"],
                "store_id": rec["store_id"],
                "country": rec["country"],
                "email": rec["email"],
            }
        )

    return texts, metadatas


def ingest_customers():
    print("Fetching customer_profile_docs_with_id...")
    texts, metadatas = build_customer_docs()
    print(f"Found {len(texts)} customer docs. Inserting into vector store...")

    if not texts:
        print("No customer docs to ingest.")
        return

    ids = customer_vs.add_texts(texts=texts, metadatas=metadatas)
    print(f"Ingested {len(ids)} customer profile vectors.")


# ---- FILM INGESTION ----

def build_film_docs():
    rows, cols = fetch_view_rows("film_profile_docs_with_id")

    texts = []
    metadatas = []

    for row in rows:
        rec = dict(zip(cols, row))

        actors = rec["actors"] or "Unknown cast"
        description = rec["description"] or "No description available"

        text = (
            f"Film profile: {rec['title']} (film_id={rec['film_id']}). "
            f"Description: {description} "
            f"Release year: {rec['release_year']}. "
            f"Rating: {rec['rating'] or 'Unrated'}. "
            f"Rental rate is {rec['rental_rate']} for a duration of {rec['rental_duration']} days. "
            f"Length: {rec['length']} minutes. Replacement cost: {rec['replacement_cost']}. "
            f"Special features: {rec['special_features'] or 'None listed'}. "
            f"Actors in this film: {actors}."
        )

        texts.append(text)
        metadatas.append(
            {
                "doc_id": str(rec["doc_id"]),
                "entity_type": "film_profile",
                "film_id": rec["film_id"],
                "title": rec["title"],
                "rating": rec["rating"],
                "release_year": rec["release_year"],
            }
        )

    return texts, metadatas


def ingest_films():
    print("Fetching film_profile_docs_with_id...")
    texts, metadatas = build_film_docs()
    print(f"Found {len(texts)} film docs. Inserting into vector store...")

    if not texts:
        print("No film docs to ingest.")
        return

    ids = film_vs.add_texts(texts=texts, metadatas=metadatas)
    print(f"Ingested {len(ids)} film profile vectors.")


# ---- ACTOR INGESTION ----

def build_actor_docs():
    rows, cols = fetch_view_rows("actor_profile_docs_with_id")

    texts = []
    metadatas = []

    for row in rows:
        rec = dict(zip(cols, row))

        full_name = f"{rec['first_name']} {rec['last_name']}"
        films = rec["films"] or "No films listed"

        text = (
            f"Actor profile: {full_name} (actor_id={rec['actor_id']}). "
            f"They appear in {rec['film_count']} films. "
            f"Filmography: {films}. "
            f"Record last updated at {rec['last_update']}."
        )

        texts.append(text)
        metadatas.append(
            {
                "doc_id": str(rec["doc_id"]),
                "entity_type": "actor_profile",
                "actor_id": rec["actor_id"],
                "actor_name": full_name,
                "film_count": rec["film_count"],
            }
        )

    return texts, metadatas


def ingest_actors():
    print("Fetching actor_profile_docs_with_id...")
    texts, metadatas = build_actor_docs()
    print(f"Found {len(texts)} actor docs. Inserting into vector store...")

    if not texts:
        print("No actor docs to ingest.")
        return

    ids = actor_vs.add_texts(texts=texts, metadatas=metadatas)
    print(f"Ingested {len(ids)} actor profile vectors.")


# ---- MAIN ENTRYPOINT ----

if __name__ == "__main__":
    print("Starting ingestion for customers, films, and actors...")
    ingest_customers()
    ingest_films()
    ingest_actors()
    print("Ingestion completed.")
