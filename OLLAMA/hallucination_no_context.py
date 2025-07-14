from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from word2number import w2n

# Load once (outside loop)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def normalize_numbers(text):
    words = text.split()
    normalized_words = []
    for word in words:
        try:
            number = w2n.word_to_num(word)
            normalized_words.append(str(number))
        except:
            normalized_words.append(word)
    return ' '.join(normalized_words)

def preprocess_text(text):
    text = ' '.join(text.lower().split())
    text = normalize_numbers(text)
    return text

def semantic_match(answer1, answer2, threshold=0.8):
    """Check if two answers are semantically similar."""
    embeddings = model.encode([answer1, answer2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold

def check_hallucination_no_context(llm_answer, correct_answer):
    llm_answer_clean = preprocess_text(llm_answer)
    correct_answer_clean = preprocess_text(correct_answer)

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
