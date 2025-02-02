from os import fsync
import time
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt
from hallucination_detection import check_hallucination

from datasets import load_dataset
squad_dataset = load_dataset('squad', split='validation')
top_5_entries = squad_dataset.select(range(5))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

client = OpenAI()

def get_llm_response(entry):
    # Get the context and question from the SQuAD entry
    context = entry['context']
    question = entry['question']
    correct_answer = entry['answers']['text'][0]  # First answer as correct answer

    # Create the messages for the chat model
    messages = [
        {"role": "system", "content": "You are a helpful assistant.Give answers only with the main words"},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
    ]
    # Call the OpenAI API to get the response
    completion = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=10,
        temperature=0
    )
    # Extract the answer from the model's response
    llm_answer = completion.choices[0].message.content.strip() 
    llm_answer = llm_answer.lower()
    correct_answer=correct_answer.lower()

    print(llm_answer,correct_answer)
    return llm_answer, correct_answer

    
def f1_score(prediction, correct_answer):
    """Calculate the F1 score between the prediction and ground truth."""
    pred_tokens = prediction.split()
    gt_tokens = correct_answer.split()
    common = set(pred_tokens) & set(gt_tokens)
    
    # Calculate precision, recall, and F1
    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common) / len(gt_tokens) if len(gt_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall, precision
    
def compare_answers(prediction, correct_answer):
   
   return f1_score(prediction, correct_answer) 


# create a model that gets the every answer of the llm and compare them and displays only the f1 score of the whole
def main():
    # Initialize counters for F1 score calculation
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
                print(f"Hallucination detected: {reason}")
            # Calculate precision, recall, and F1
            precision, recall, f1 = f1_score(llm_answer, correct_answer)
            
        
            
            # Accumulate metrics
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_samples += 1
            pbar.update(1)
    # Record end time
    end_time = time.time()

    # Calculate averages
    avg_precision = total_precision / num_samples if num_samples > 0 else 0
    avg_recall = total_recall / num_samples if num_samples > 0 else 0
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    elapsed_time = end_time - start_time

    # Display results
    print(f" - Average Precision: {avg_precision:.4f}")
    print(f" - Average Recall:    {avg_recall:.4f}")
    print(f" - Average F1 Score:  {avg_f1:.4f}")
    print(f"Time Taken: {elapsed_time:.2f} seconds")
    print(f"Hallucinations detected : {hallucination_count}")


if __name__ == "__main__":
    main()