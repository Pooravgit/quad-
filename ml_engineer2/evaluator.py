from ingest import load_documents
from chunk_ember import chunk_documents
from vector_store import build_faiss_index, search_faiss_index
from llm_reasoning import run_llm_on_query
from sentence_transformers import SentenceTransformer, util

# Queries and ground truth justifications
queries = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

ground_truth_answers = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

def compare_justifications(gen_text, gt_text, model):
    embedding1 = model.encode(gen_text, convert_to_tensor=True)
    embedding2 = model.encode(gt_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

def main():
    folder_path = "docs"
    model = SentenceTransformer('all-MiniLM-L6-v2')
    generated_answers = {}

    # Step 1: Load and index documents
    docs = load_documents(folder_path)
    chunked_docs = chunk_documents(docs)
    build_faiss_index(chunked_docs)

    # Step 2: Loop over each query and process sequentially
    for idx, query in enumerate(queries):
        print(f"\nQuery {idx+1}: {query}")

        results = search_faiss_index(query)
        response = run_llm_on_query(query, results)

        # Extract justification (assumes response is a dict with a "justification" key)
        justification = response.get("justification", "").strip()
        ground_truth = ground_truth_answers[idx]

        # Compare using semantic similarity
        similarity_score = compare_justifications(justification, ground_truth, model)

        # Store result
        generated_answers[query] = {
            "justification": justification,
            "ground_truth": ground_truth,
            "similarity": round(similarity_score, 4)
        }

        # Print inline comparison
        print(f"Generated Justification:\n{justification}")
        print(f"Ground Truth:\n{ground_truth}")
        print(f"Similarity Score: {similarity_score:.4f}")

    return generated_answers

if __name__ == "__main__":
    results = main()
