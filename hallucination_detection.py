from os import fsync
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()
modelb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
modela = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text
    
def check_hallucination(llm_answer, context, correct_answer):
    # Preprocess answers and context
    llm_answer = preprocess_text(llm_answer)
    correct_answer = preprocess_text(correct_answer)
    context = preprocess_text(context)
    
    # Exact match check
    if llm_answer == correct_answer:
        return False, "No hallucination detected"

    embeddingsb = modelb.encode([llm_answer, context])
    similarityb = cosine_similarity([embeddingsb[0]], [embeddingsb[1]])[0][0]
    if similarityb < 0.6:  # Adjust threshold
        return True, "Semantic mismatch-general"

    return False, "No hallucination detected"

    # Semantic similarity (if needed for additional checks)
    #embeddingsa = modela.encode([llm_answer, context])
    #similaritya = cosine_similarity([embeddingsa[0]], [embeddingsa[1]])[0][0]
    #if similaritya < 0.8:  # Adjust threshold
    #    return True, "Semantic mismatch multi-qa"