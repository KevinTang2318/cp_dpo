import re
import json
from sklearn.metrics import accuracy_score


def calculate_accuracy(
        pattern: str,
        file_path: str,
        ground_truth_key: str,
        llm_answer_key: str,
        need_extraction: bool = False
):

    with open(file_path, "r") as f:
        data = json.load(f)

    # get the ground truth and answer from the LLM
    ground_truth = []
    llm_answers = []

    for entry in data:
        if need_extraction:
            entry[ground_truth_key] = entry[ground_truth_key][0]

        cleaned_ground_truth = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '',
                                      entry[ground_truth_key])
        ground_truth.append(cleaned_ground_truth.upper())

        match = re.match(pattern, entry[llm_answer_key])
        if match:
            # Extract just the letter (A, B, C, or D)
            llm_answer = match.group()
            cleaned_llm_answer = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '',
                                        llm_answer)
            llm_answers.append(cleaned_llm_answer.upper())
        else:
            raise Exception("Failed to match: ", entry[llm_answer_key])

    return accuracy_score(ground_truth, llm_answers)


print("Accuracy for zero-shot CP on AQuA dataset: ",
      calculate_accuracy(
          "^[A-E]",
          "preference_data/aqua_preference_dataset.json",
          "ground_truth",
          "correct_answer"
      )
      )

print("Accuracy for zero-shot CP on StrategyQA dataset: ",
      calculate_accuracy(
          "^(Yes|No)",
          "preference_data/strategy_qa_preference_dataset.json",
          "ground_truth",
          "correct_answer"
      )
      )

print("Accuracy for zero-shot CP on CoinFlip dataset: ",
      calculate_accuracy(
          "^(Yes|No)",
          "preference_data/coin_flip_preference_dataset.json",
          "rationale",
          "correct_answer"
      )
      )

print("Accuracy for zero-shot CP on BigBench Object Tracking dataset: ",
      calculate_accuracy(
          "^.*$",
          "preference_data/object_tracking_preference_dataset.json",
          "rationale",
          "correct_answer",
          need_extraction=True
      )
      )
