from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once (outside loop)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def clean_text(text):
    return ' '.join(text.lower().strip().split())

def semantic_match(answer1, answer2, threshold=0.8):
    """Check if two answers are semantically similar."""
    embeddings = model.encode([answer1, answer2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold

def check_hallucination_no_context(llm_answer, correct_answer):
    llm_answer_clean = clean_text(llm_answer)
    correct_answer_clean = clean_text(correct_answer)

    # Exact match (after normalization)
    if llm_answer_clean == correct_answer_clean:
        return False, "No hallucination detected (Exact match)"

    # Fuzzy match
    fuzzy_score = fuzz.ratio(llm_answer_clean, correct_answer_clean)
    if fuzzy_score > 85:
        return False, f"No hallucination detected (Fuzzy match: {fuzzy_score}%)"

    # Semantic match
    if semantic_match(llm_answer_clean, correct_answer_clean):
        return False, "No hallucination detected (Semantic similarity)"

    return True, "Hallucination detected: LLM answer does not match or align semantically with reference"
