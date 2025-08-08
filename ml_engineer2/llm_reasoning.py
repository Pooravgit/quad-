# llm_reasoning.py
import os
import json
from typing import List, Dict

import google.generativeai as genai

# Google/GenAI settings
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


def build_prompt(question: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = (
        "You are an expert assistant analyzing legal or policy documents."
        " Based on the following chunks, answer the question in the specified structured JSON format.\n\n"
        "Required JSON Output:\n"
        "{\n"
        '  \"decision\": \"Yes\" | \"No\" | null,\n'
        '  \"justification\": \"Your detailed explanation based on clause references\",\n'
        '  \"used_clauses\": [\"Clause x\", \"Clause y\"]\n'
        "}\n\n"
        "Instructions:\n"
        "- Only use information explicitly present in the chunks.\n"
        "- Do not assume facts that are not stated in the context.\n"
        "- If the question can be answered as a Yes/No decision, include it in the `decision` field.\n"
        "- If the question is descriptive, set `decision` to null.\n\n"
        f"Question: {question}\n\n"
        f"Context Chunks:\n{context}"
    )
    return prompt


def query_llm(question: str, faiss_results: List[Dict]) -> Dict:
    """
    Returns a parsed JSON response from the model (or an error dict).
    """

    if not GOOGLE_API_KEY:
        return {"error": "GOOGLE_API_KEY not set. Set GOOGLE_API_KEY in env variables to use Gemini."}

    # extract the texts
    retrieved_texts = []
    for doc in faiss_results:
        # doc may be a SimpleNamespace (.page_content) or dict
        if hasattr(doc, "page_content"):
            retrieved_texts.append(doc.page_content)
        elif isinstance(doc, dict) and "page_content" in doc:
            retrieved_texts.append(doc["page_content"])
        else:
            # fallback: string representation
            retrieved_texts.append(str(doc))

    prompt = build_prompt(question, retrieved_texts)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(
            [
                {"role": "system", "content": "You are a legal assistant AI for policy analysis."},
                {"role": "user", "content": prompt}
            ],
            generation_config={"response_mime_type": "application/json"}
        )

        # try to get raw text from different possible attributes
        raw_reply = None
        if hasattr(response, "text"):
            raw_reply = response.text
        elif hasattr(response, "candidates") and len(response.candidates) > 0:
            # candidate content may be present
            cand = response.candidates[0]
            raw_reply = getattr(cand, "content", str(cand))
        else:
            raw_reply = str(response)

        raw_reply = raw_reply.strip()
        try:
            return json.loads(raw_reply)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON produced by model", "raw": raw_reply}

    except Exception as e:
        return {"error": str(e)}


def run_llm_on_query(question: str, faiss_results: List[Dict]) -> Dict:
    return query_llm(question, faiss_results)
