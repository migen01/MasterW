from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    #Normalize text
    return ' '.join(text.lower().split())

def check_hallucination(llm_answer, context, correct_answer):
    
    llm_answer = preprocess_text(llm_answer)
    correct_answer = preprocess_text(correct_answer)
    context = preprocess_text(context)

    #print(f"\n Checking Hallucination for: {llm_answer}")
    #print(f" Correct Answer: {correct_answer}")
    #print(f" Context: {context}")

    # Check if LLM answer directly matches correct answer
    fuzzy_score = fuzz.ratio(llm_answer, correct_answer) / 100.0  # Normalize fuzzy score
    #print(f" Fuzzy Match Score: {fuzzy_score:.2f}")

    if fuzzy_score > 0.75:  
        #print(" No hallucination (Fuzzy Match)")
        return False, "No hallucination detected (Fuzzy match)"

    # Step 2: Check if LLM answer is present in context (semantic match)
    embeddingsb = modelb.encode([llm_answer, context])
    similarityb = cosine_similarity([embeddingsb[0]], [embeddingsb[1]])[0][0]
    #print(f" Semantic Similarity Score: {similarityb:.2f}")

    if similarityb >= 0.50:  
        #print("No hallucination (Semantic Match)")
        return False, "No hallucination detected (Semantic match)"

    #print(" Hallucination detected!")
    return True, "Semantic mismatch - possible hallucination"
