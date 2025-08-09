from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

stemmer = PorterStemmer()
def f1_score(prediction, correct_answer):
    #Calculate the F1 score between the prediction and ground truth
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

def logic_f1_score(prediction, correct_answer, context):
    #Calculate the adjusted F1 score considering constraints
    f1, recall, precision = f1_score(prediction, correct_answer)
    penalty = check_constrains(prediction, context)
    adjusted_f1 = f1 - penalty * 0.1
    return max(adjusted_f1, 0), recall, precision


def calculate_precision_recall_f1(correct_answer, inferred_facts):
    
    correct_tokens = set(stemmer.stem(word) for word in word_tokenize(correct_answer.lower()))
    found_tokens = set()

    for fact in inferred_facts:
        fact_parts = [str(part).lower().strip() for part in fact]
        for part in fact_parts:
            found_tokens.update(set(stemmer.stem(word) for word in word_tokenize(part)))

    if not found_tokens:
        return 0.0, 0.0, 0.0  # If no answer was found, Precision, Recall, and F1 score are 0.

    common_tokens = correct_tokens & found_tokens  # Correct words present in both

    precision = len(common_tokens) / len(found_tokens) if found_tokens else 0
    recall = len(common_tokens) / len(correct_tokens) if correct_tokens else 0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def answer_found_f1(correct_answer, inferred_facts, context):
    #Checks if the correct answer appears in any part of the inferred facts
    if not inferred_facts:
        return False, 0.0, 0.0, 0.0  # Ensure all values are floats

    correct_answer = correct_answer.lower().strip()
    
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0
    #found_match = False

    for fact in inferred_facts:
        fact_text = " ".join(map(str, fact))
        f1, recall, precision = logic_f1_score(fact_text, correct_answer, context)

        if f1 > best_f1:
            best_f1 = f1
            best_recall = recall
            best_precision = precision
            #found_match = True

    return  best_precision, best_recall
