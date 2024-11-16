import json
import re


def generate_preference_dataset_for_aqua(file_path):
    preference_data = []
    failed_to_match = []
    with open(file_path, 'r') as f:
        samples = json.load(f)

    pattern = r"Correct answer:\s*(.*?)\s*;\s*Wrong answer:\s*(.*?)\s*;\s*(.+)"
    successful_parses = 0
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
                "correct_answer": correct_answer,
                "wrong_answer": wrong_answer,
                "llm_reasoning": reasoning
            })

        else:
            failed_to_match.append(sample)

    with open('preference_data/aqua_preference_dataset.json', 'w') as f:
        json.dump(preference_data, f, indent=4)

    print(
        f"Successfully parsed {successful_parses} out of {len(samples)} samples")

    return failed_to_match


def generate_preference_dataset_for_strategy_qa(file_path):
    preference_data = []
    failed_to_match = []
    with open(file_path, 'r') as f:
        samples = json.load(f)

    pattern = r"Correct answer:\s*(.*?)\s*;\s*Wrong answer:\s*(.*?)\s*;(?:\s*(.+))?"
    successful_parses = 0
    for sample in samples:
        matches = re.match(pattern, sample['llama_output'], re.DOTALL)

        if matches:
            successful_parses += 1
            correct_answer = matches.group(1).strip()
            wrong_answer = matches.group(2).strip()
            reasoning = matches.group(3).strip() if matches.group(3) else ""

            preference_data.append({
                "question": sample['input'],
                "options": ["Yes", "No"],
                "rationale": sample['target'],
                "correct_answer": correct_answer,
                "wrong_answer": wrong_answer,
                "llm_reasoning": reasoning
            })

        else:
            failed_to_match.append(sample)

    with open('preference_data/strategy_qa_preference_dataset.json', 'w') as f:
        json.dump(preference_data, f, indent=4)

    print(
        f"Successfully parsed {successful_parses} out of {len(samples)} samples")

    return failed_to_match


if __name__ == '__main__':
    # parsed all data
    failed_to_match = generate_preference_dataset_for_aqua(
        'llm_output/aqua_output.json'
    )

    print("AQuA failed responses: ")
    for failed_samples in failed_to_match:
        print(failed_samples)
        print("---------------------------------------------")

    # parsed 2285/2290 data. The rest failed due to questions contain
    # sensitive information and llm refused to answer.
    failed_to_match = generate_preference_dataset_for_strategy_qa(
        'llm_output/strategy_qa_output.json'
    )
    print("StrategyQA failed responses: ")
    for failed_samples in failed_to_match:
        print(failed_samples)
        print("---------------------------------------------")
