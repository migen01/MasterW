import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from os import fsync
import time
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

client = OpenAI()

squad_dataset = load_dataset('squad', split='validation')

top_5_entries = squad_dataset.select(range(20))

# Logical Neural Network for QA
class LogicalNeuralNetworkQA(nn.Module):
    def __init__(self, llm_name="gpt2"):
        super(LogicalNeuralNetworkQA, self).__init__()
    
    # Neural network layers
        self.input_layer = nn.Linear(10, 16)  # Example: 10 input features for structured knowledge
        self.hidden_layer = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)  # Binary classification or score prediction

    # Load a pre-trained LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)

    # Set the padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def forward(self, question, knowledge_facts, logical_rules):
        knowledge_text = '; '.join([' '.join(map(str, fact)) for fact in knowledge_facts])
        logical_scores = []

        for rule in logical_rules:
            input_text = f"Question: {question}\nKnowledge: {knowledge_text}\nRule: {rule}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # Create attention mask
            output = self.llm.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50  # Generate only new tokens
            )
            logical_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            score = len(logical_response) / 100.0  # Example scoring heuristic
            logical_scores.append(score)

        logical_output = torch.tensor(logical_scores).mean()  # Aggregate logical scores
        combined_output = logical_output  # Assuming simple combination for now
        return combined_output


# Helper functions
def get_llm_response(entry):
    context = entry['context']
    question = entry['question']
    correct_answer = entry['answers']['text'][0]  # First answer as correct answer

    messages = [
        {"role": "system", "content": "You are a helpful assistant.Give answers only with the main words"},
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=10,
        temperature=0
    )

    llm_answer = completion.choices[0].message.content.strip()
    return llm_answer, correct_answer


def f1_score(prediction, ground_truth):
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    common = set(pred_tokens) & set(gt_tokens)

    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common) / len(gt_tokens) if len(gt_tokens) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, recall, precision


def main():
    total_f1 = 0
    total_f1_lnn = 0
    num_samples = 0

    start_time = time.time()

    dataset_subset = top_5_entries
    total_examples = len(dataset_subset)

    # Initialize the Logical Neural Network
    lnn_qa = LogicalNeuralNetworkQA()

    print("Processing dataset...")
    for entry in tqdm(dataset_subset, total=total_examples, desc="Progress", unit="examples"):
        llm_answer, correct_answer = get_llm_response(entry)

        # Calculate F1 score for plain LLM
        f1, recall, precision = f1_score(llm_answer, correct_answer)
        total_f1 += f1

        # Simulate LNN input and rules (dummy values for knowledge and rules)
        knowledge_facts = [[1] * 10]  # Example: 10 features
        logical_rules = ["If X is a parent of Y, then Y cannot be a parent of X."]
        lnn_output = lnn_qa(entry['question'], knowledge_facts, logical_rules)

        # Calculate F1 score for LNN prediction (assuming scalar prediction here)
        lnn_prediction = str(lnn_output.item())  # Convert output to string for F1 calculation
        f1_lnn, _, _ = f1_score(lnn_prediction, correct_answer)
        total_f1_lnn += f1_lnn

        num_samples += 1

    end_time = time.time()

    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    avg_f1_lnn = total_f1_lnn / num_samples if num_samples > 0 else 0
    elapsed_time = end_time - start_time

    print(f" - Average F1 Score (LLM only): {avg_f1:.4f}")
    print(f" - Average F1 Score (LNN):       {avg_f1_lnn:.4f}")
    print(f"Time Taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
