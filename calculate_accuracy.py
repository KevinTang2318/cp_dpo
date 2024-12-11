import json
import re

import pandas as pd
from sklearn.metrics import accuracy_score


def extract_answer(entry, pattern):
    match = re.match(pattern, entry)
    if match:
        # Extract just the letter (A, B, C, or D)
        llm_answer = match.group()
        return llm_answer.upper()
    else:
        raise Exception("Failed to match: ", entry)


def calculate_accuracy(
        pattern: str,
        dataset_name: str,
        need_extraction: bool = False
):

    test_data = pd.read_json("test_data/testing.jsonl", lines=True)

    dataset_entries = (test_data[test_data["dataset"]
                                 == f"{dataset_name}_preference_dataset.json"]
                       .reset_index(drop=True))

    # get the ground truth and answer from the LLM
    ground_truth = dataset_entries["ground_truth"].replace(
        r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', regex=True).str.upper().tolist()

    llm_answers = dataset_entries["response_chosen"].apply(
        extract_answer, args=(pattern,)
    ).replace(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', regex=True).str.upper().tolist()

    return accuracy_score(ground_truth, llm_answers)


print("Accuracy for zero-shot CP on AQuA dataset: ",
      calculate_accuracy(
          "^[A-E]",
          "aqua",
      )
      )

print("Accuracy for zero-shot CP on StrategyQA dataset: ",
      calculate_accuracy(
          "^(Yes|No)",
          "strategy_qa",
      )
      )

print("Accuracy for zero-shot CP on CoinFlip dataset: ",
      calculate_accuracy(
          "^(Yes|No)",
          "coin_flip",
      )
      )

print("Accuracy for zero-shot CP on BigBench Object Tracking dataset: ",
      calculate_accuracy(
          "^.*$",
          "object_tracking",
      )
      )
