import argparse
import json
import re
from calculate_accuracy import extract_answer


def clean_text(text):
    return re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", text)


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
            "bigbench_date",
            "all"
        ],
        help="Specify one or more tasks to run inference on."
    )
    args = parser.parse_args()

    answer_patterns = {
        "aqua": r"([A-E])",
        "strategy_qa": "",
        "coin_flip": "",
        "last_letter": "",
        "bigbench_date": "",
        "object_tracking": ""
    }

    if len(args.tasks) == 1 and args.tasks[0] == "all":
        tasks = [
            "aqua",
            "strategy_qa",
            "coin_flip",
            "object_tracking",
            "last_letter",
            "bigbench_date"
        ]
    else:
        tasks = args.tasks

    for task in tasks:
        for dataset in ["train", "val"]:
            with open(
                f"preference_data/{task}_{dataset}_preference_dataset.json",
                'r'
            ) as f:
                samples = json.load(f)

            correct_answer_count = 0
            wrong_answer_count = 0
            for sample in samples:
                if isinstance(sample["ground_truth"], list):
                    sample["ground_truth"] = sample["ground_truth"][0]

                ground_truth = clean_text(sample["ground_truth"].upper())
                if answer_patterns[task] == "":
                    correct_answer = clean_text(
                        sample["correct_answer"].upper())
                    wrong_answer = clean_text(
                        sample["wrong_answer"].upper())
                else:
                    correct_answer = clean_text(extract_answer(
                        sample["correct_answer"], answer_patterns[task]))
                    wrong_answer = clean_text(extract_answer(
                        sample["wrong_answer"], answer_patterns[task]))

                correct_answer_count += 1 if ground_truth == correct_answer else 0
                wrong_answer_count += 1 if ground_truth != wrong_answer else 0

            print(f"{task} {dataset} statistics:")
            print("Correct answer matches ground truth:",
                  f"{correct_answer_count}/{len(samples)}")
            print("Wrong answer deviates from ground truth:",
                  f"{wrong_answer_count}/{len(samples)}")
