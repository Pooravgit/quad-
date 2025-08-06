import requests
import json
# llm_reasoning.py
import requests
import json
from typing import List, Dict
import ollama

# Customize this based on where Ollama is running
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2" 


def ask_ollama_llm(prompt, model=MODEL):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    data = {"model": model,"prompt": prompt,"stream": False}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()['response'].strip()
    else:
        raise Exception(f"Ollama request failed: {response.text}")
    
def build_prompt(question: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
    
    prompt = (
        "You are an assistant analyzing policy documents. Based on the following chunks, answer the question in structured JSON format:\n\n"
        "Required JSON format:\n"
        "{\n"
        '  "decision": "Yes" or "No",\n'
        '  "justification": "Your explanation",\n'
        '  "used_clauses": ["Clause x", "Clause y"]\n'
        "}\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )
    
    return prompt
    
MODEL = "llama 3.2"


def query_llm(question: str, faiss_results: List[Dict]) -> Dict:
    retrieved_texts = [doc.page_content for doc in faiss_results]
    prompt = build_prompt(question, retrieved_texts)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a legal assistant AI for policy analysis."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        response.raise_for_status()

        result = response.json()
        raw_reply = result['message']['content']

        try:
            return json.loads(raw_reply)
        except json.JSONDecodeError:
            print("Couldn't parse JSON output. Raw output:\n", raw_reply)
            return {"error": "Invalid JSON format", "raw": raw_reply}

    except Exception as e:
        print("Error calling local LLM:", str(e))
        return {"error": str(e)}


user_query = "Can the policyholder terminate the policy early due to job relocation?"
llm_response = query_llm(user_query, results)
print("Structured Output:\n", llm_response)
