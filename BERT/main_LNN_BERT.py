import torch
from os import fsync
import time
from tqdm import tqdm
from hallucination_no_context import check_hallucination_no_context
from hallucination_context import check_hallucination
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset

from LNN_BERT import RuleBasedLogicalNetwork

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from fuzzywuzzy import fuzz
nltk.download('punkt_tab')
stemmer = PorterStemmer()

def load_squad_dataset(num_samples):
    #squad_dataset = load_dataset("squad", split="validation")
    squad_dataset = load_dataset('squad_v2',split='validation')
    raw_top_entries = squad_dataset.select(range(num_samples))
    filtered_entries = raw_top_entries.filter(lambda example: example.get('answers', {}).get('text', []))
    return filtered_entries

dataset = load_squad_dataset(217) #for squadv2 - 217 is 100 examples(because of filtering)

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
model.eval()  # set to evaluation mode

##https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/roberta#transformers.RobertaForQuestionAnswering
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
#model = RobertaForQuestionAnswering.from_pretrained("FacebookAI/roberta-base")

def preprocess_text(text):
    text = text.lower()# Lowercase
    text = ' '.join(text.split())# Remove whitespaces
    return text

def extract_fact_from_qa(question, answer):
    """
    Heuristic-based triple generation from question/answer.
    You can expand this with dependency parsing or use OpenIE/Spacy for better accuracy.
    """
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

# Load BERT
def get_llm_response(entry): #  
    context = entry['context']
    question = entry['question']
    c_ans = entry.get('answers', {}).get('text', [])
    correct_answer = c_ans[0] if c_ans else ""

    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most probable answer span
    answer_start = torch.argmax(outputs.start_logits) # check meaning
    answer_end = torch.argmax(outputs.end_logits) + 1 #check 

    input_ids = inputs["input_ids"][0]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    answer = preprocess_text(answer)
    correct_answer = correct_answer.lower().strip()

    #print(f"Generated answer (BERT): {answer}", f"Correct: {correct_answer}")
    return answer, correct_answer



def calculate_f1_score(correct_answer, found_answer):
    #Computes the F1 score using token overlap
    
    correct_tokens = set(stemmer.stem(word) for word in word_tokenize(correct_answer.lower()))
    found_tokens = set(stemmer.stem(word) for word in word_tokenize(found_answer.lower()))

    if not found_tokens:
        return 0.0  # If no answer was found, F1 score is 0.

   
    common_tokens = correct_tokens & found_tokens  # Correct words present in both

    precision = len(common_tokens) / len(found_tokens) if found_tokens else 0
    recall = len(common_tokens) / len(correct_tokens) if correct_tokens else 0

    if precision + recall == 0:
        return 0.0  

    f1 = (2 * precision * recall) / (precision + recall)
    return f1



for entry in dataset:
    generated_answer, correct_answer = get_llm_response(entry)
    f1_score_value = calculate_f1_score(correct_answer, generated_answer)
    #print(f"F1 Score (BERT QA): {f1_score_value:.2f}")
    


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

logic_engine = RuleBasedLogicalNetwork()

def main():
   
    
    hallucination_count = 0
    total_f1_score = 0.0
    total_examples = 0

    start_time = time.time()

    with tqdm(total=len(dataset), desc="Processing") as pbar:
        for entry in dataset:
            logic_engine = RuleBasedLogicalNetwork()  

            generated_answer, correct_answer = get_llm_response(entry)
            f1_score = calculate_f1_score(correct_answer, generated_answer)

            # Convert into a fact (subject, predicate, object)
            fact = extract_fact_from_qa(entry['question'], generated_answer)
            logic_engine.add_fact(fact)

            # Run inference
            logic_engine.infer()

            # Query based on correct answer
            query = (None, None, correct_answer.lower())
            results = logic_engine.query_pattern(query)

            context = entry['context']
            
            # Check for hallucinations
            hallucination, reason = check_hallucination_no_context(generated_answer ,correct_answer)
            #hallucination, reason = check_hallucination(generated_answer, context ,correct_answer)
            if hallucination:
                print(f"Generated answer (BERT): {generated_answer}", f"Correct: {correct_answer}") 
                print(f"Hallucination Detected: {hallucination}, Reason: {reason}\n")
                hallucination_count += 1

            #is_correct = len(results) > 0
            #hallucination = not is_correct

            #print(f"\nQ: {entry['question']}")
            #print(f"A: {generated_answer} | Correct: {correct_answer}")
            #print(f"Inferred Facts: {results}")
            #print(f"F1 Score: {f1_score:.2f}")

            total_f1_score += f1_score
            total_examples += 1
            pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time 

    avg_f1 = total_f1_score / total_examples if total_examples > 0 else 0

    print(f"\nAverage F1 Score: {avg_f1:.2f}")
    print(f"Time Taken: {elapsed_time:.2f} seconds")
    print(f" - Total Hallucinations:         {hallucination_count}")

if __name__ == "__main__":
    main()
