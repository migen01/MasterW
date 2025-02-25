from os import fsync
import time
from openai import OpenAI
from tqdm import tqdm
from llm_hallucination_detection import check_hallucination

from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')
top_5_entries = squad_dataset.select(range(100))

def preprocess_text(text):
    text = text.lower()# Lowercase
    text = ' '.join(text.split())# Remove whitespaces
    return text

client = OpenAI()

def get_llm_response(entry):
    context = entry['context']
    question = entry['question']
    correct_answer = entry['answers']['text'][0]  # First answer as correct answer

    messages = [
        {"role": "system", "content": "You are a helpful assistant.Give answers only with the main words"}, #chat model message
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    
    llm_answer = completion.choices[0].message.content.strip() 
    llm_answer = llm_answer.lower()
    correct_answer=correct_answer.lower() # Extract answer 

    #print(llm_answer,correct_answer)
    return llm_answer, correct_answer

def f1_score(prediction, correct_answer):
    #Calculate F1 score
    pred_tokens = prediction.split()
    gt_tokens = correct_answer.split()
    common = set(pred_tokens) & set(gt_tokens)

    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common) / len(gt_tokens) if len(gt_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall, precision

def non_empty_constrains(answer):
    return 1 if len(answer.strip()) ==0 else 0

def context_word_constraint(answer, context):
    context_words = set(context.split())
    answer_words = set(answer.split())
    return 1 if not answer_words.issubset(context_words) else 0 

def check_constrains(answer, context):
    penalty = 0 
    penalty += non_empty_constrains(answer)
    penalty += context_word_constraint(answer, context)
    return penalty

def compare_answers(prediction, correct_answer):
   
   return f1_score(prediction, correct_answer) 

def logic_f1_score(prediction, correct_answer, context):
    f1 ,recall, precision = f1_score(prediction, correct_answer)
    penalty = check_constrains(prediction, context)
    adjusted_f1 = f1 - penalty * 0.1
    return max(adjusted_f1, 0), recall, precision


# create a model that gets the every answer of the llm to compare them and displays only the f1 score of the whole
def main():
    total_f1 = 0
    num_samples = 0
    total_precision = 0 
    total_recall = 0
    hallucination_count = 0

    start_time = time.time()

    dataset_subset = top_5_entries
    total_examples = len(dataset_subset)

    print("Processing dataset...")
    with tqdm(total=total_examples, desc="Progress", unit="examples") as pbar:
        for entry in dataset_subset:
            llm_answer, correct_answer = get_llm_response(entry)
            
            # Check for hallucinations
            hallucination, reason = check_hallucination(llm_answer, entry['context'], correct_answer)
            if hallucination:
                hallucination_count += 1
                print(f"Hallucination detected(LLM): {reason}")
            
            adjusted_f1, recall ,precision = logic_f1_score(llm_answer, correct_answer, entry['context'])
            #precision, recall, f1 = f1_score(llm_answer, correct_answer)
                
            # metrics
            total_precision += precision
            total_recall += recall
            total_f1 += adjusted_f1 
            num_samples += 1
            pbar.update(1)

    end_time = time.time()

    # averages
    avg_precision = total_precision / num_samples if num_samples > 0 else 0
    avg_recall = total_recall / num_samples if num_samples > 0 else 0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    elapsed_time = end_time - start_time

    print(f" - Average Precision: {avg_precision:.4f}")
    print(f" - Average Recall:    {avg_recall:.4f}")
    print(f" - Average F1 Score:  {avg_f1:.4f}")
    print(f"Time Taken: {elapsed_time:.2f} seconds")
    print(f"Hallucinations detected : {hallucination_count}")


if __name__ == "__main__":
    main()