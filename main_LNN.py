from datasets import load_dataset
import json 
import time
from openai import OpenAI
from tqdm import tqdm

from lnn_hallucination_detection import check_hallucination
from LNN import RuleBasedLogicalNetwork
from F1_LNN import answer_found_f1

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from fuzzywuzzy import fuzz

nltk.download('punkt_tab')
stemmer = PorterStemmer()
client = OpenAI()

def load_squad_dataset(num_samples=20):
    squad_dataset = load_dataset("squad", split="validation")
    dataset_subset = squad_dataset.select(range(num_samples))
    return dataset_subset

def extract_knowledge_from_squad_entry(entry):
    
    #Extracts structured facts and rules from database
    context = entry["context"]
    question = entry["question"]
    #correct_answer = entry["answers"]["text"][0].lower() if entry["answers"]["text"] else "unknown"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured knowledge from the provided context. "
                    "Ensure that key facts are **standalone statements** and avoid complex phrasing.\n\n"
                    "Correct Output Example:\n"
                    "{\n"
                    "  \"facts\": [\n"
                    "    [\"Super Bowl 50\", \"was_played_at\", \"Levi's Stadium\"],\n"
                    "    [\"Levi's Stadium\", \"is_in\", \"Santa Clara, California\"],\n"
                    "    [\"Super Bowl 50\", \"was_played_in\", \"Santa Clara, California\"]\n"
                    "  ]\n"
                    "}\n\n"
                    "Incorrect Output:\n"
                    "{\n"
                    "  \"facts\": [\n"
                    "    [\"Super Bowl 50\", \"was_played_in\", \"San Francisco Bay Area, Santa Clara, California\"]\n"
                    "  ]\n"
                    "}\n\n"
                    "⚠️ Return only valid JSON, no extra text."
                )
            },
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nExtract:"}
        ]
    )

    response_text = response.choices[0].message.content.strip()

    try:
        data = json.loads(response_text)  # Ensure JSON format
    except json.JSONDecodeError:
       # print(" Error: Invalid JSON. Response received:\n", response_text)
        return {"rules": [], "facts": []}

    #print("\n Extracted Facts:", json.dumps(data.get("facts", []), indent=2))
    return data

def make_hashable(item):
    # converts lists into tuples to make them hashable
    if isinstance(item, list):
        return tuple(make_hashable(subitem) for subitem in item)  # Recursively handle nested lists
    return item 

def reasoning_with_squad(entry):
    #Uses dataset entries to extract facts, apply reasoning, and infer new knowledge.
    data = extract_knowledge_from_squad_entry(entry)
    lnn = RuleBasedLogicalNetwork()

   # Convert facts into hashable tuples
    known_facts = {make_hashable(fact) for fact in data['facts']}
    for fact in known_facts:
        lnn.add_fact(fact)

    #print("\nKnown Facts Before Inference:", lnn.known_facts)
    location_facts = {fact[0]: fact[2] for fact in known_facts if len(fact) == 3 and fact[1] == "is_in"}
    event_locations = [(fact[0], fact[2]) for fact in known_facts if len(fact) == 3 and fact[1] == "was_played_at"] # the verbs were taken as examples
 
    # inference Rules
    inferred_facts = {
        (event, "was_played_in", location_facts[venue])
        for event, venue in event_locations if venue in location_facts
    }

    for inferred_fact in inferred_facts:
       # print(f"✅ Inferring: {inferred_fact}")
        lnn.add_fact(inferred_fact)

    # Run inference
    lnn.infer()

    #print("\nKnown Facts After Inference:", lnn.known_facts)
    return lnn.known_facts


from difflib import get_close_matches

def query_squad_reasoning(entry, query_pattern):
   
    inferred_facts = reasoning_with_squad(entry)

    matched_facts = list(inferred_facts) 

    return matched_facts


def calculate_f1_score(correct_answer, found_answer):
    #Computes the F1 score using token overlap
    
    correct_tokens = set(stemmer.stem(word) for word in word_tokenize(correct_answer.lower()))
    found_tokens = set(stemmer.stem(word) for word in word_tokenize(found_answer.lower()))

    if not found_tokens:
        return 0.0  # If no answer was found, F1 score is 0.

    # Calculate Precision, Recall, and F1 Score
    common_tokens = correct_tokens & found_tokens  # Correct words present in both

    precision = len(common_tokens) / len(found_tokens) if found_tokens else 0
    recall = len(common_tokens) / len(correct_tokens) if correct_tokens else 0

    if precision + recall == 0:
        return 0.0  

    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def answer_found(correct_answer, inferred_facts):
    #Checks if the correct answer appears in any part of the inferred facts and calculates F1 score with better similarity matching.
    
    if not inferred_facts:
        return False, 0.0  # Ensure the second value is a float

    correct_answer = correct_answer.lower().strip()
    best_f1 = 0.0
    found_match = False
    best_found_answer = ""

    for fact in inferred_facts:
        fact_parts = [str(part).lower().strip() for part in fact]

        for part in fact_parts:
            f1 = calculate_f1_score(correct_answer, part)
            fuzzy_score = fuzz.ratio(correct_answer, part) / 100.0 

            combined_score = (f1 + fuzzy_score) / 2  # Weighted score

            if combined_score > best_f1:
                best_f1 = combined_score
                best_found_answer = part

            if correct_answer in part:
                found_match = True  # match found

   # print(f" Best Found Answer: {best_found_answer} with Improved Score: {best_f1:.2f}")

    return found_match, float(best_f1)  

def main():
    dataset = load_squad_dataset(100)  
    hallucination_count = 0
    total_f1_score = 0.0  
    total_examples = 0  
    start_time = time.time()
    total_precision = 0.0
    total_recall = 0.0

    print("Processing SQuAD dataset...")
    with tqdm(total=len(dataset), desc="Progress", unit="examples") as pbar:
        for entry in dataset:
            correct_answer = entry["answers"]["text"][0] if entry["answers"]["text"] else "unknown"

            query_patterns = [
                (correct_answer, None, None),  # Answer as subject
                (None, correct_answer, None),  # Answer as relation
                (None, None, correct_answer),  # Answer as object
                (None, "represents", correct_answer),  # Common relationship
                ("Super Bowl 50", None, correct_answer)  # If answer is related to event
            ]

            # Store results from multiple queries
            results = []
            for query in query_patterns:
                results.extend(query_squad_reasoning(entry, query))

            # Remove duplicates 
            results = list(set(results))

            #Extract the best-matching answer and compute F1 score
            is_correct,f1_score_value = answer_found(correct_answer, results)
            #print(f"DEBUG: is_correct: {is_correct}, f1_score_value: {f1_score_value}, type: {type(f1_score_value)}")
            context = entry["context"]
            recall, precision = answer_found_f1(correct_answer, results, context)
            
            # Update F1 score 
            total_precision += precision
            total_recall += recall

            total_f1_score += (f1_score_value) 
            total_examples += 1 

            # Convert results into a readable string
            inferred_text = " ".join([" ".join(map(str, fact)) for fact in results]) if results else "No inference available"
            
            hallucination, reason = check_hallucination(inferred_text, entry['context'], correct_answer)

            if hallucination:
                hallucination_count += 1
               # print(f" Hallucination detected (LLM): {reason}")

            print(f"\n Question: {entry['question']}")
            print(f" Correct Answer: {correct_answer}")
            print(f" Inference Result: {results[:3]}")
            print(f" Answer Found in Inference: {' Yes' if is_correct else ' No'}")
            print(f" F1 Score: {f1_score_value:.2f}")
            print(f"Hallucinations detected: {hallucination_count}")

            pbar.update(1)

    end_time = time.time()

    average_precision = total_precision / total_examples if total_examples > 0 else 0
    average_recall = total_recall / total_examples if total_examples > 0 else 0
    average_f1 = total_f1_score / total_examples if total_examples > 0 else 0
    elapsed_time = end_time - start_time

    print("\nFinal Results:")
    print(f" Average Precision: {average_precision:.2f}")
    print(f" Average Recall: {average_recall:.2f}")
    print(f" Average F1 Score: {average_f1:.2f}")
    print(f"Time Taken: {elapsed_time:.2f} seconds")
    print(f" Total Hallucinations Detected: {hallucination_count}/{total_examples} examples")


if __name__ == "__main__":
    main()
