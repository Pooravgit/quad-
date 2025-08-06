from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
INDEX_PATH = "faiss_index"

def build_faiss_index(chunked_docs):
    texts = [chunk["text"] for doc in chunked_docs for chunk in doc["chunks"]]
    metadatas = [
        {
            "source": chunk["source"],
            "type": chunk["type"],
            "doc_chunk_id": chunk["doc_chunk_id"]
        }
        for doc in chunked_docs for chunk in doc["chunks"]
    ]
    db = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    db.save_local(INDEX_PATH)
    print("FAISS index saved.")

def search_faiss_index(query: str, k: int = 5):
    db = FAISS.load_local(INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
    results = db.similarity_search(query, k=k)
    return results
