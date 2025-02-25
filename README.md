<<<<<<< HEAD
# MasterW

"export OPENAI_API_KEY='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"

This script evaluates the performance of a large language model (LLM) using the SQuAD validation dataset. It calculates metrics such as precision, recall, 
and F1 score to compare the model's predictions against the ground truth answers provided in the dataset.

Overview

The script performs the following tasks:

Dataset Loading: Loads a subset of the SQuAD validation dataset using the Hugging Face datasets library.
LLM Integration: Queries an LLM (using OpenAI's API) to answer questions from the dataset based on the provided context.
Answer Comparison: Compares the LLM-generated answers with the ground truth answers using precision, recall, and F1 score.
Metrics Aggregation: Computes and displays average precision, recall, and F1 score across all examples.
Performance Reporting: Outputs the evaluation metrics and the time taken for processing.

Key Components

Functions
    get_llm_response(entry) - Takes a SQuAD dataset entry and queries the LLM for an answer.

Inputs:
context: The context passage from the dataset.
question: The question related to the context.
correct_answer: The first ground truth answer.
Outputs:
llm_answer: The response generated by the LLM.
correct_answer: The ground truth answer.

    f1_score(prediction, ground_truth) - Computes the F1 score, precision, and recall for a given pair of predicted and ground truth answers.

Inputs:
prediction: The LLM's answer.
ground_truth: The actual answer.
Outputs:
f1: The F1 score.
recall: The recall metric.
precision: The precision metric.

    compare_answers(prediction, ground_truth) - A helper function to calculate and return the F1 score for a prediction.

Main Execution Flow
Dataset Subset: Selects a small subset (top 85 entries) of the SQuAD dataset for evaluation.
Model Evaluation: Iterates through each dataset entry, retrieves the LLM's response, and calculates metrics.
Result Aggregation: Aggregates precision, recall, and F1 scores across all examples.
Performance Reporting: Outputs average metrics and total processing time.
Usage

Run the script as follows:

    "python main.py"

After processing, the script displays:

Average Precision: Measures the proportion of relevant words in the prediction.
Average Recall: Measures the proportion of relevant words from the ground truth captured in the prediction.
Average F1 Score: Harmonic mean of precision and recall.
Time Taken: Total time taken to process the dataset subset.
=======
Neuro Symbolic AI
>>>>>>> origin/main
