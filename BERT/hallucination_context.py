from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from word2number import w2n

# Load once
modelb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

def semantic_match(a1, a2, threshold=0.85):
    embeddings = modelb.encode([a1, a2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold

def check_hallucination(llm_answer, context, correct_answer):
    llm_answer = preprocess_text(llm_answer)
    correct_answer = preprocess_text(correct_answer)
    context = preprocess_text(context)

    #  Exact match
    if llm_answer == correct_answer:
        return False, "No hallucination (Exact match)"

    # Fuzzy string match
    fuzzy_score = fuzz.ratio(llm_answer, correct_answer)
    if fuzzy_score > 92:
        return False, f"No hallucination (Fuzzy match: {fuzzy_score}%)"

    # Semantic match between generated and correct answer
    if semantic_match(llm_answer, correct_answer):
        return False, "No hallucination (Semantic similarity to correct answer)"

    # Check if LLM hallucinated something not even in context
    context_sim = semantic_match(llm_answer, context, threshold=0.6)
    if context_sim:
        return False, "No hallucination (Semantic alignment with context)"

    return True, "Hallucination detected: No sufficient semantic or textual alignment"
