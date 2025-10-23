import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from openai import OpenAI
import streamlit as st
import numpy as np
import uuid

# ==========================
# 1. Load API Keys
# ==========================
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ==========================
# 2. Initialize Clients
# ==========================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# 3. Create Collection
# ==========================
def create_collection(collection_name="imdb_movies", vector_size=1536):
    """Membuat collection di Qdrant"""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{collection_name}' berhasil dibuat di Qdrant.")


# ==========================
# 4. Load Dataset
# ==========================
def load_data(csv_path="imdb_top_1000.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Overview"])  # pastikan Overview tidak kosong
    return df


# ==========================
# 5. Generate Embeddings
# ==========================
def get_embedding(text):
    """Membuat embedding dari teks menggunakan OpenAI model"""
    text = text.replace("\n", " ")
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ==========================
# 6. Insert Data ke Qdrant
# ==========================
def insert_data(collection_name="imdb_movies", csv_path="imdb_top_1000.csv", limit=200):
    """Membaca dataset, membuat embedding, dan insert ke Qdrant"""
    df = load_data(csv_path).head(limit)
    points = []

    print(f"📦 Membuat embedding dan upload {len(df)} data ke Qdrant...")
    for i, row in df.iterrows():
        movie_id = str(uuid.uuid4())
        overview = row["Overview"]
        vector = get_embedding(overview)

        payload = {
            "title": row["Series_Title"],
            "year": row["Released_Year"],
            "genre": row["Genre"],
            "rating": row["IMDB_Rating"],
            "overview": overview
        }

        points.append(PointStruct(id=movie_id, vector=vector, payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    print("✅ Data berhasil diinsert ke Qdrant!")


# ==========================
# 7. Cek Hasil Insert
# ==========================
def test_search(collection_name="imdb_movies", query="space adventure"):
    """Test pencarian berdasarkan query teks"""
    query_vector = get_embedding(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5
    )

    print("\n🔍 Hasil pencarian:")
    for result in search_result:
        payload = result.payload
        print(f"- {payload['title']} ({payload['year']}) | {payload['genre']} | Score: {result.score:.3f}")


# ==========================
# 8. Main Test
# ==========================
if __name__ == "__main__":
    create_collection()
    insert_data(limit=50)
    test_search(query="romantic love story")
