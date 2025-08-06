from ingest import load_documents
from chunk_ember import chunk_documents
from vector_store import build_and_query_vector_store

def main():
    folder_path = "docs"

    # Step 1: Load documents
    docs = load_documents(folder_path)
    print(f" Loaded {len(docs)} documents.")

    # Step 2: Chunk documents
    chunked_docs = chunk_documents(docs)
    total_chunks = sum(len(doc["chunks"]) for doc in chunked_docs)
    print(f" Chunked into {total_chunks} total chunks.")

    # Step 3: Build vector store and run sample query
    build_and_query_vector_store(chunked_docs)

if __name__ == "__main__":
    main()
