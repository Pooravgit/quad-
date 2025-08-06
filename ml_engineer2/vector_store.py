from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load your documents and chunk them
from ingest import load_documents
from chunk_ember import chunk_documents

docs = load_documents("docs")
chunked_docs = chunk_documents(docs)

# Prepare texts and metadata
texts = [chunk["text"] for doc in chunked_docs for chunk in doc["chunks"]]
metadatas = [chunk for doc in chunked_docs for chunk in doc["chunks"]]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and save FAISS index
db = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadatas)
db.save_local("faiss_index")

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
