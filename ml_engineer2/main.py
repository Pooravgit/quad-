from ingest import load_documents
from chunk_ember import chunk_documents
from vector_store import build_faiss_index, search_faiss_index
from llm_reasoning import run_llm_on_query

def main():
    folder_path = "docs"

    # Step 1: Load documents
    docs = load_documents(folder_path)
    print(f"Loaded {len(docs)} documents.")

    # Step 2: Chunk documents
    chunked_docs = chunk_documents(docs)
    total_chunks = sum(len(doc["chunks"]) for doc in chunked_docs)
    print(f"Chunked into {total_chunks} total chunks.")

    # Step 3: Build FAISS vector store
    build_faiss_index(chunked_docs)

    # Step 4: Define query and search in vector store
    query = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
    results = search_faiss_index(query)
    print(f"Retrieved {len(results)} relevant chunks.")
    print(query)

    # Step 5: Send to LLM for reasoning
    llm_response = run_llm_on_query(query, results)
    print("\nLLM Structured Output:\n", llm_response)

if __name__ == "__main__":
    main()
