# chunk_ember.py
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs: List[dict], chunk_size=400, chunk_overlap=64):
    """
    Splits each document's text into chunks for embedding/storage.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    for doc in docs:
        text = doc.get("full_text", "")
        if not text:
            doc["chunks"] = []
            continue
        chunks = splitter.split_text(text)
        doc["chunks"] = [
            {
                "text": chunk,
                "source": doc["source"],
                "type": doc["type"],
                "doc_chunk_id": f"{doc['source']}_chunk_{i}"
            }
            for i, chunk in enumerate(chunks)
        ]
    return docs
