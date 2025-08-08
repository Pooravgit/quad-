# vector_store.py
import os
import psycopg2
from pgvector.psycopg2 import register_vector, Vector
from sentence_transformers import SentenceTransformer
import numpy as np
from types import SimpleNamespace

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set.")
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def build_faiss_index(chunked_docs):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            doc_chunk_id TEXT UNIQUE,
            source TEXT,
            type TEXT,
            content TEXT,
            embedding vector({EMBEDDING_DIM})
        );
    """)
    conn.commit()

    for doc in chunked_docs:
        for chunk in doc.get("chunks", []):
            emb = model.encode(chunk["text"], convert_to_numpy=True)
            cur.execute("""
                INSERT INTO embeddings (doc_chunk_id, source, type, content, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_chunk_id) DO UPDATE
                SET content=EXCLUDED.content, embedding=EXCLUDED.embedding,
                    source=EXCLUDED.source, type=EXCLUDED.type;
            """, (chunk["doc_chunk_id"], chunk["source"], chunk["type"],
                  chunk["text"], Vector(emb.tolist())))
    conn.commit()
    cur.close()
    conn.close()
    print("Data stored in PostgreSQL vector store.")

def search_faiss_index(query: str, k: int = 5):
    conn = get_conn()
    cur = conn.cursor()
    q_emb = model.encode(query, convert_to_numpy=True)
    cur.execute("""
        SELECT content, source, type, doc_chunk_id
        FROM embeddings
        ORDER BY embedding <-> %s
        LIMIT %s;
    """, (Vector(q_emb.tolist()), k))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [SimpleNamespace(page_content=row[0], metadata={"source": row[1], "type": row[2], "doc_chunk_id": row[3]})
            for row in rows]
