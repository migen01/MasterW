import time
from tqdm import tqdm
from datasets import load_dataset
from hallucination_context import check_hallucination
from hallucination_no_context import check_hallucination_no_context
from LNN_Gemini import RuleBasedLogicalNetwork  # assumed defined
from fuzzywuzzy import fuzz
import google.generativeai as genai
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
stemmer = PorterStemmer()

# Load Gemini API
genai.configure(api_key="AIzaSyDPCFe2bnYno6Uu7FnZQffI4SBk3Md6f2Q")
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Load SQuAD
squad_dataset = load_dataset('squad', split='validation')
#squad_dataset = load_dataset('squad_v2',split='validation')
raw_top_entries = squad_dataset.select(range(100)) #for squadv2 - 217 is 100 examples(because of filtering)
dataset = raw_top_entries.filter(lambda example: example.get('answers', {}).get('text', []))


def preprocess_text(text):
    return ' '.join(text.lower().split())

def get_llm_response(entry):
    context = entry['context']
    question = entry['question']
    c_ans = entry.get('answers', {}).get('text', [])
    correct_answer = c_ans[0] if c_ans else ""

    prompt = f"""You are a helpful assistant. Give answers only with the main words.
Context: {context}
Question: {question}
Answer:"""

    response = model.generate_content(prompt)
    generated_answer = response.text.strip().lower()
    correct_answer = correct_answer.strip().lower()
    #print(f"Generated: {generated_answer} | Correct: {correct_answer}")
    return generated_answer, correct_answer

def extract_fact_from_qa(question, answer):
    question = question.lower()
    answer = answer.strip().lower()

    if "who" in question:
        return [answer, "is", None]
    elif "when" in question:
        return [None, "happened_on", answer]
    elif "where" in question:
        return [None, "located_in", answer]
    elif "what" in question:
        return [None, "is", answer]
    else:
        return [None, "relates_to", answer]

def calculate_f1_score(correct_answer, found_answer):
    correct_tokens = set(stemmer.stem(w) for w in word_tokenize(correct_answer))
    found_tokens = set(stemmer.stem(w) for w in word_tokenize(found_answer))

    if not found_tokens:
        return 0.0

    common_tokens = correct_tokens & found_tokens
    precision = len(common_tokens) / len(found_tokens)
    recall = len(common_tokens) / len(correct_tokens)

    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def answer_found(correct_answer, inferred_facts):
    if not inferred_facts:
        return False, 0.0

    correct_answer = correct_answer.lower().strip()
    best_f1 = 0.0
    found_match = False

    for fact in inferred_facts:
        fact_parts = [str(part).lower().strip() for part in fact]

        for part in fact_parts:
            f1 = calculate_f1_score(correct_answer, part)
            fuzzy_score = fuzz.ratio(correct_answer, part) / 100.0
            combined_score = (f1 + fuzzy_score) / 2

            if combined_score > best_f1:
                best_f1 = combined_score
            if correct_answer in part:
                found_match = True

    return found_match, float(best_f1)

def main():
    hallucination_count = 0
    total_f1_score = 0.0
    total_examples = 0

    start_time = time.time()

    with tqdm(total=len(dataset), desc="Processing") as pbar:
        for entry in dataset:
            logic_engine = RuleBasedLogicalNetwork()  # Reset per question

            generated_answer, correct_answer = get_llm_response(entry)
            f1_score = calculate_f1_score(correct_answer, generated_answer)

            fact = extract_fact_from_qa(entry['question'], generated_answer)
            logic_engine.add_fact(fact)
            logic_engine.infer()

            query = (None, None, correct_answer.lower())
            results = logic_engine.query_pattern(query)
            
            hallucination, reason = check_hallucination_no_context(generated_answer,correct_answer)
            #hallucination, reason = check_hallucination(generated_answer, entry['context'],correct_answer)
            if hallucination:
                #print(f"Q: {entry['question']}")
                #print(f"Generated: {generated_answer} | Correct_: {correct_answer}")
                print(f"Hallucination: {hallucination} — Reason: {reason}\n")
                hallucination_count += 1

            total_f1_score += f1_score
            total_examples += 1
            pbar.update(1)

    end_time = time.time()
    avg_f1 = total_f1_score / total_examples if total_examples > 0 else 0

    print(f"\nAverage F1 Score: {avg_f1:.2f}")
    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Total Hallucinations: {hallucination_count}")

if __name__ == "__main__":
    main()
