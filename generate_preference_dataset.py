import json
import re


def generate_preference_dataset_for_aqua(train_path, val_path):
    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = r"Correct answer:\s*(.*?)\s*;\s*Wrong answer:\s*(.*?)\s*;\s*(.+)"
    successful_parses = 0
    preference_data = []
    failed_to_match = []

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1)
                wrong_answer = matches.group(2)
                reasoning = matches.group(3)

                preference_data.append({
                    "question": sample['question'],
                    "options": sample['options'],
                    "rationale": sample['rationale'],
                    "ground_truth": sample['correct'],
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
            f"preference_data/aqua_{dataset_type}_preference_dataset.json", 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


def generate_preference_dataset_for_strategy_qa(train_path, val_path):

    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = r"Correct answer:\s*(.*?)\s*;\s*Wrong answer:\s*(.*?)\s*;(?:\s*(.+))?"
    successful_parses = 0
    preference_data = []
    failed_to_match = []

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1).strip()
                wrong_answer = matches.group(2).strip()
                reasoning = matches.group(
                    3).strip() if matches.group(3) else ""

                ground_truth = None
                for key, value in sample['target_scores'].items():
                    if value == 1:
                        ground_truth = key
                        break

                preference_data.append({
                    "question": sample['input'],
                    "options": ["Yes", "No"],
                    "rationale": sample['target'],
                    "ground_truth": ground_truth,
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
                f"preference_data/strategy_qa_{dataset_type}_preference_dataset.json", 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


def generate_preference_dataset_for_object_tracking(train_path, val_path):

    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\;\s*"
        r"Wrong answer:\s*(.*?)\;\s*"  # Match "Wrong answer" until the period
    )
    pattern_double = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\.\s*"
        r"Wrong answer:\s*(.*?)\.\s*"  # Match "Wrong answer" until the period
    )
    pattern_three = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\.\s*"
        r"Wrong answer:\s*(.*?)\;\s*"  # Match "Wrong answer" until the period
    )

    successful_parses = 0
    preference_data = []
    failed_to_match = []

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)
            if not matches:
                matches = re.match(
                    pattern_double, sample['llama_output'], re.DOTALL)
            if not matches:
                matches = re.match(
                    pattern_three, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1).strip()
                wrong_answer = matches.group(2).strip()
                reasoning_start = matches.end()  # Start of the reasoning text
                # Get everything after the matched section
                reasoning = sample['llama_output'][reasoning_start:].strip()

                preference_data.append({
                    "question": sample['input'],
                    "options": [item for item in sample["target_scores"]],
                    "ground_truth": [item for item in sample["target_scores"] if sample["target_scores"][item] == 1],
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
            f'preference_data/object_tracking_{dataset_type}_preference_dataset.json', 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


def generate_preference_dataset_for_coin_flip(train_path, val_path):
    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = r"Correct answer:\s*(.*?)\s*;\s*Wrong answer:\s*(.*?)\s*;(\s*(.+))?"

    preference_data = []
    failed_to_match = []
    successful_parses = 0

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1).strip()
                wrong_answer = matches.group(2).strip()
                reasoning = matches.group(
                    3).strip() if matches.group(3) else ""

                preference_data.append({
                    "question": sample['question'],
                    "options": ["Yes", "No"],
                    "ground_truth": sample['answer'],
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
            f'preference_data/coin_flip_{dataset_type}_preference_dataset.json', 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


def generate_preference_dataset_for_last_letter(train_path, val_path):

    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\;\s*"
        r"Wrong answer:\s*(.*?)\;\s*"  # Match "Wrong answer" until the period
    )
    pattern_double = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\.\s*"
        r"Wrong answer:\s*(.*?)\.\s*"  # Match "Wrong answer" until the period
    )
    pattern_three = (
        # Match "Correct answer" until the period
        r"Correct answer:\s*(.*?)\.\s*"
        r"Wrong answer:\s*(.*?)\;\s*"  # Match "Wrong answer" until the period
    )

    successful_parses = 0
    preference_data = []
    failed_to_match = []

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)
            if not matches:
                matches = re.match(
                    pattern_double, sample['llama_output'], re.DOTALL)
            if not matches:
                matches = re.match(
                    pattern_three, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1).strip()
                wrong_answer = matches.group(2).strip()
                reasoning_start = matches.end()  # Start of the reasoning text
                # Get everything after the matched section
                reasoning = sample['llama_output'][reasoning_start:].strip()

                preference_data.append({
                    "question": sample['question'],
                    "options": [],
                    "ground_truth": sample['answer'],
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
            f'preference_data/last_letter_{dataset_type}_preference_dataset.json', 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


def generate_preference_dataset_for_bigbench_date(train_path, val_path):

    with open(train_path, 'r') as f:
        train_samples = json.load(f)

    with open(val_path, 'r') as f:
        val_samples = json.load(f)
    datasets = [("train", train_samples), ("val", val_samples)]

    pattern = "Correct answer:\s*(.*?);\s*Wrong answer:\s*(.*?);(.*)?"

    successful_parses = 0
    preference_data = []
    failed_to_match = []

    for dataset_type, samples in datasets:
        for sample in samples:
            matches = re.match(pattern, sample['llama_output'], re.DOTALL)

            if matches:
                successful_parses += 1
                correct_answer = matches.group(1).strip()
                wrong_answer = matches.group(2).strip()
                reasoning = matches.group(
                    3).strip() if matches.group(3) else ""

                preference_data.append({
                    "question": sample['input'],
                    "options": [item for item in sample["target_scores"]],
                    "ground_truth": [item for item in sample["target_scores"] if sample["target_scores"][item] == 1],
                    "correct_answer": correct_answer,
                    "wrong_answer": wrong_answer,
                    "llm_reasoning": reasoning
                })

            else:
                failed_to_match.append(sample)

        with open(
            f'preference_data/bigbench_date_{dataset_type}_preference_dataset.json', 'w'
        ) as f:
            json.dump(preference_data, f, indent=4)

        print("*************************************")
        print(
            f"Successfully parsed {successful_parses} out of {len(samples)} samples")
        for failed_samples in failed_to_match:
            print(failed_samples)
            print("---------------------------------------------")

        successful_parses = 0
        preference_data = []
        failed_to_match = []

    return failed_to_match


if __name__ == '__main__':
    # parsed all data
    generate_preference_dataset_for_aqua(
        'llm_output/aqua_train_output.json',
        'llm_output/aqua_val_output.json'
    )

    generate_preference_dataset_for_coin_flip(
        'llm_output/coin_flip_train_output.json',
        'llm_output/coin_flip_val_output.json'
    )

    generate_preference_dataset_for_object_tracking(
        "llm_output/object_tracking_train_output.json",
        "llm_output/object_tracking_val_output.json"
    )

    generate_preference_dataset_for_strategy_qa(
        "llm_output/strategy_qa_train_output.json",
        "llm_output/strategy_qa_val_output.json"
    )

    generate_preference_dataset_for_last_letter(
        "llm_output/last_letter_train_output.json",
        "llm_output/last_letter_val_output.json"
    )

    generate_preference_dataset_for_bigbench_date(
        "llm_output/bigbench_date_train_output.json",
        "llm_output/bigbench_date_val_output.json"
    )
