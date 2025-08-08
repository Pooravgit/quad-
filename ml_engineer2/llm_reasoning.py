import requests
import json
from typing import List, Dict
import google.generativeai as genai

# Google API settings
MODEL_NAME = "gemini-1.5-flash"  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

# def build_prompt(question: str, retrieved_chunks: List[str]) -> str:
#     context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

#     prompt = (
#         "You are an assistant analyzing policy documents. Based on the following chunks, answer the question in structured JSON format:\n\n"
#         "Required JSON format:\n"
#         "{\n"
#         '  "decision": "Yes" or "No",\n'
#         '  "justification": "Your explanation",\n'
#         '  "used_clauses": ["Clause x", "Clause y"]\n'
#         "}\n\n"
#         f"Question: {question}\n\n"
#         f"Context:\n{context}"
#     )

#     return prompt

def build_prompt(question: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = (
        "You are an expert assistant analyzing legal or policy documents."
        "Based on the following chunks, "
        "answer the question in the specified structured JSON format.\n\n"
        
        "Required JSON Output:\n"
        "{\n"
        '  "decision": "Yes" | "No" | NULL,\n'
        '  "justification": "Your detailed explanation based on clause references",\n'
        '  "used_clauses": ["Clause x", "Clause y", ...]\n'
        "}\n\n"

        " Instructions:\n"
        "- Only use information explicitly present in the chunks.\n"
        "- Do not assume facts that are not stated in the context."
        "- If the question can be answered as a Yes/No decision, include it in the `decision` field.\n"
        "- If the policy does not **explicitly** state inclusion or exclusion, default to 'No' for decision."
        "- First, determine if the question is asking whether something is allowed/covered/included/excluded. If YES, then it is a decision question and return a Yes/No in `decision`.\n"
        "- If the question is descriptive (e.g., 'What is covered...?', 'Explain clause...', 'What does the policy say about...'), set `decision` to null.\n"
        "- If not applicable (e.g., descriptive or informational questions), set `decision` to null.\n"
        "- The `justification` should be **clear and concise**, ideally 2–3 sentences.\n"
        "- Avoid overly legalistic or repetitive language in the justification.\n"
        "- Always include a clear, clause-based justification.\n"
        "- Mention which clauses were referred to while answering.\n\n"

        f"Now analyze the following:\n"
        f"Question: {question}\n\n"
        f"Context Chunks:\n{context}"
    )

    return prompt

def query_llm(question: str, faiss_results: List[Dict]) -> Dict:
    retrieved_texts = [doc.page_content for doc in faiss_results]
    prompt = build_prompt(question, retrieved_texts)

    try:
        model = genai.GenerativeModel(MODEL_NAME)

        response = model.generate_content(
            [
                {"role": "system", "parts": "You are a legal assistant AI for policy analysis."},
                {"role": "user", "parts": prompt}
            ],
            generation_config={
                "response_mime_type": "application/json"
            }
        )

#         response = requests.post(
#             OLLAMA_URL,
#             json={
#                 "model": MODEL_NAME,
#                 "messages": [
#                     {"role": "system", "content": "You are a legal assistant AI for policy analysis."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "stream": False
#             }
#         )
#         response.raise_for_status()

        raw_reply = response.text.strip()

        try:
            return json.loads(raw_reply)
        except json.JSONDecodeError:
            print("Couldn't parse JSON output. Raw output:\n", raw_reply)
            return {"error": "Invalid JSON format", "raw": raw_reply}

    except Exception as e:
        print("Error calling Gemini API:", str(e))
        return {"error": str(e)}


def run_llm_on_query(question: str, faiss_results: List[Dict]) -> Dict:
    return query_llm(question, faiss_results)
