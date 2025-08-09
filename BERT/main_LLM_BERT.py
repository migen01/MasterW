
import torch
from os import fsync
import time
from tqdm import tqdm
from hallucination_no_context import check_hallucination_no_context
from hallucination_context import check_hallucination


from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from datasets import load_dataset

def load_squad_dataset(num_samples):
    #squad_dataset = load_dataset("squad", split="validation")
    squad_dataset = load_dataset('squad_v2',split='validation')
    raw_top_entries = squad_dataset.select(range(num_samples))
    filtered_entries = raw_top_entries.filter(lambda example: example.get('answers', {}).get('text', []))
    return filtered_entries

dataset = load_squad_dataset(217)  #for squadv2 - 217 is 100 examples(because of filtering)

def preprocess_text(text):
    text = text.lower()
    text = ' '.join(text.split())
    return text

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")

##https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/roberta#transformers.RobertaForQuestionAnswering
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
#model = RobertaForQuestionAnswering.from_pretrained("FacebookAI/roberta-base")


def get_llm_response(entry):
    context = entry['context']
    question = entry['question']
    c_ans = entry.get('answers', {}).get('text', [])
    correct_answer = c_ans[0] if c_ans else ""
    
##https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/bert#transformers.BertForQuestionAnswering
    
    #tokenizer to make them as a pair of text
    inputs = tokenizer(question, context, return_tensors="pt",truncation=True,
    padding=True,              
    return_attention_mask=True 
    )
    with torch.no_grad():
        outputs = model(**inputs)

##https://huggingface.co/docs/transformers/v4.52.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput

    # Get the most probable answer span
    answer_start = torch.argmax(outputs.start_logits) 
    answer_end = torch.argmax(outputs.end_logits) + 1 #check 

    input_ids = inputs["input_ids"][0]
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    answer = preprocess_text(answer)
    correct_answer = correct_answer.lower().strip()

    #print(f"Generated answer (BERT): {answer}", f"Correct: {correct_answer}") 
    return answer, correct_answer

from collections import Counter

def f1_score(prediction, correct_answer):
    pred_tokens = prediction.split()
    gt_tokens = correct_answer.split()
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    
    common = pred_counts & gt_counts  # intersection: min count per token
    num_common = sum(common.values())

    precision = num_common / len(pred_tokens) if pred_tokens else 0
    recall = num_common / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
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

def logic_f1_score(prediction, correct_answer, context):
    f1, recall, precision = f1_score(prediction, correct_answer)
    penalty = check_constrains(prediction, context)
    adjusted_f1 = f1 - penalty * 0.1
    return max(adjusted_f1, 0), recall, precision

# create a model that gets the every answer of the llm to compare them and displays only the f1 score of the whole

def main():
    total_f1 = 0
    num_samples = 0
    hallucination_count = 0

    start_time = time.time()


    with tqdm(total=len(dataset), desc="Processing") as pbar:
        for entry in dataset:
            llm_answer, correct_answer = get_llm_response(entry)
            context = entry['context']

            # hallucination detection function
            hallucination, reason = check_hallucination_no_context(llm_answer ,correct_answer)
            #hallucination, reason = check_hallucination(llm_answer, context, correct_answer)
            if hallucination:
                print(f"Generated answer (BERT): {llm_answer}", f"Correct: {correct_answer}") 
                print(f"Hallucination Detected: {hallucination}, Reason: {reason}\n")
                hallucination_count += 1

            # Compute F1 Score (quality of answer compared to reference)
            adjusted_f1, _, _ = logic_f1_score(llm_answer, correct_answer, context)
            total_f1 += adjusted_f1
            num_samples += 1

            pbar.update(1)

    end_time = time.time()

    # Compute averages
    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0
    elapsed_time = end_time - start_time

    print(f" - Average F1 Score:               {avg_f1:.4f}")
    print(f" - Time Taken:                    {elapsed_time:.2f} seconds")
    print(f" - Total Hallucinations:          {hallucination_count}")


if __name__ == "__main__":
    main()