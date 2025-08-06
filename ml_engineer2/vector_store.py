# vector_store.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_and_query_vector_store(chunked_docs, query="What is covered under coronary heart disease?"):
    # Step 1: Prepare texts and metadata
    texts = [chunk["text"] for doc in chunked_docs for chunk in doc["chunks"]]
    metadatas = [
        {
            "source": chunk["source"],
            "type": chunk["type"],
            "doc_chunk_id": chunk["doc_chunk_id"]
        }
        for doc in chunked_docs for chunk in doc["chunks"]
    ]

    # Step 2: Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Step 3: Create FAISS index
    db = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
    db.save_local("faiss_index")
    print(" FAISS index saved.")

    # Step 4: Load FAISS index
    db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

    # Step 5: Search
    results = db.similarity_search(query, k=5)

<<<<<<< HEAD
    print("\n Query Results:")
    for i, res in enumerate(results):
        print(f"\nResult #{i+1}")
        print("Text:", res.page_content[:300])
        print("Metadata:", res.metadata)
=======
# Load saved FAISS index with secure override
db = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# Sample user query
query = "What is covered under amputation or chemotherapy?"

# Search similar chunks
results = db.similarity_search(query, k=5)

# Print results
for i, res in enumerate(results):
    print(f"\nResult #{i+1}")
    print("Text:", res.page_content)
    print("Metadata:", res.metadata)
>>>>>>> 9ce40775907351dbdbb4e270d05eb32983ad791a
