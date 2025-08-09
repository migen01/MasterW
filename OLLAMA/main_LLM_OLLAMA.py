import os
import time
import requests
from tqdm import tqdm
from datasets import load_dataset
from hallucination_context import check_hallucination 
from hallucination_no_context import check_hallucination_no_context

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3"  # Replace with whatever model you've pulled

# Load SQuAD dataset
#squad_dataset = load_dataset('squad_v2', split='validation')
squad_dataset = load_dataset('squad', split='validation')
raw_top_entries = squad_dataset.select(range(100))
all_entries = raw_top_entries.filter(lambda example: example.get('answers', {}).get('text', []))


def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    return text


def get_llm_response(entry):
    context = entry['context']
    question = entry['question']
    c_ans = entry.get('answers', {}).get('text', [])
    correct_answer = c_ans[0] if c_ans else ""

    prompt = f"""You are a helpful assistant. Give answers only with the main words.
Context: {context}
Question: {question}
Answer:"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        llm_answer = response.json().get("response", "").strip().lower()
    except requests.RequestException as e:
        print("Error calling Ollama:", e)
        llm_answer = ""

    correct_answer = correct_answer.lower()

    print(f"Generated answer: {llm_answer}", f"Correct: {correct_answer}")
    return llm_answer, correct_answer


def f1_score(prediction, correct_answer):
    pred_tokens = prediction.split()
    gt_tokens = correct_answer.split()
    common = set(pred_tokens) & set(gt_tokens)

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1, recall, precision


def non_empty_constrains(answer):
    return 1 if len(answer.strip()) == 0 else 0


def context_word_constraint(answer, context):
    context_words = set(context.split())
    answer_words = set(answer.split())
    return 1 if not answer_words.issubset(context_words) else 0


def check_constrains(answer, context):
    penalty = 0
    penalty += non_empty_constrains(answer)
    penalty += context_word_constraint(answer, context)
    return penalty


def logic_f1_score(prediction, correct_answer, context):
    f1, recall, precision = f1_score(prediction, correct_answer)
    penalty = check_constrains(prediction, context)
    adjusted_f1 = f1 - penalty * 0.1
    return max(adjusted_f1, 0), recall, precision


def main():
    total_f1 = 0
    num_samples = 0
    total_precision = 0
    total_recall = 0
    hallucination_count = 0

    start_time = time.time()

    dataset_subset = all_entries
    total_examples = len(dataset_subset)

    print("Processing dataset...\n")
    with tqdm(total=total_examples, desc="Progress", unit="examples") as pbar:
        for entry in dataset_subset:
            llm_answer, correct_answer = get_llm_response(entry)

            hallucination, reason = check_hallucination(llm_answer, entry['context'], correct_answer)
            #hallucination, reason = check_hallucination_no_context(llm_answer, correct_answer)
            if hallucination:
               # print(f"Generated: {llm_answer} | Correct_: {correct_answer}")
                print(f"Hallucination: {hallucination} — Reason: {reason}\n")
                hallucination_count += 1

            adjusted_f1, recall, precision = logic_f1_score(llm_answer, correct_answer, entry['context'])

            total_precision += precision
            total_recall += recall
            total_f1 += adjusted_f1
            num_samples += 1
            pbar.update(1)

    end_time = time.time()

    avg_precision = total_precision / num_samples if num_samples > 0 else 0
    avg_recall = total_recall / num_samples if num_samples > 0 else 0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    elapsed_time = end_time - start_time

    print(f" - Average Precision: {avg_precision:.4f}")
    print(f" - Average Recall:    {avg_recall:.4f}")
    print(f" - Average F1 Score:  {avg_f1:.4f}")
    print(f" - Total Hallucinations:         {hallucination_count}")
    print(f" - Time Taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
