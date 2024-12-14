import argparse
import json
import re

from sklearn.metrics import accuracy_score


def extract_answer(entry, pattern):
    match = re.match(pattern, entry)
    if match:
        # Extract just the letter (A, B, C, or D)
        llm_answer = match.group(1)
        return llm_answer.upper()
    else:
        raise Exception("Failed to match: ", entry)


def calculate_accuracy(
        pattern: str,
        task: str,
        ground_truth_key: str
):

    with open(f"llm_output/{task}_test_output.json", 'r') as f:
        test_samples = json.load(f)

    ground_truth = []
    predictions = []

    for sample in test_samples:
        ground_truth.append(sample[ground_truth_key].upper())
        predictions.append(extract_answer(sample["llama_output"], pattern))

    return accuracy_score(ground_truth, predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference on selected tasks.")
    parser.add_argument(
        "tasks",
        nargs="+",
        choices=[
            "aqua",
            "strategy_qa",
            "coin_flip",
            "object_tracking",
            "last_letter",
            "bigbench_date"
        ],
        help="Specify one or more tasks to run inference on."
    )
    args = parser.parse_args()

    # data preprocessing for certain datasets
    for task in ["strategy_qa", "object_tracking", "bigbench_date"]:
        with open(f"llm_output/{task}_test_output.json", 'r') as f:
            test_samples = json.load(f)

        for sample in test_samples:
            sample["ground_truth"] = [
                key for key in sample["target_scores"] if sample["target_scores"][key] == 1][0]

            if task == "object_tracking":
                sample["ground_truth"] = re.sub(
                    r'[^a-zA-Z ]', '', sample["ground_truth"])

        with open(f"llm_output/{task}_test_output.json", 'w') as f:
            json.dump(test_samples, f, indent=4)

    configs = {
        "aqua": {
            "pattern": "Answer:\s*([A-E])\)",
            "ground_truth_key": "correct"
        },
        "strategy_qa": {
            "pattern": "Answer:\s*(YES|NO)",
            "ground_truth_key": "ground_truth"
        },
        "coin_flip": {
            "pattern": "Answer:\s*(YES|NO)",
            "ground_truth_key": "answer"
        },
        "last_letter": {
            "pattern": "Answer:\s*(.*?);",
            "ground_truth_key": "answer"
        },
        "object_tracking": {
            "pattern": "Answer:\s*(.*?);",
            "ground_truth_key": "ground_truth"
        },
        "bigbench_date": {
            "pattern": "Answer:\s*(.*?);",
            "ground_truth_key": "ground_truth"
        }
    }

    for task in args.tasks:
        print(f"Accuracy for {task} dataset: ", calculate_accuracy(
            configs[task]["pattern"],
            task,
            configs[task]["ground_truth_key"]
        ))
