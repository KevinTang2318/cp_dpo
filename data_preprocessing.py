import argparse
import json
import os
import transformers
import torch
from huggingface_hub import login
from sklearn.model_selection import train_test_split

from data_loader import *
from llama3_inference import *


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

    print(f"Working on the following tasks: {args.tasks}")

    # load the datasets
    datasets = {
        "aqua": {
            "raw_data": load_aqua_data("datasets/AQuA/test.json"),
            "prompt_generator": generate_aqua_prompt,
            "contrastive_prompt_generator": generate_aqua_constrastive_prompt,
        },
        "strategy_qa": {
            "raw_data": load_strategy_qa_data("datasets/StrategyQA/task.json"),
            "prompt_generator": generate_strategy_qa_prompt,
            "contrastive_prompt_generator": generate_strategy_qa_constrastive_prompt,
        },
        "coin_flip": {
            "raw_data": load_coin_flip_data(
                "datasets/CoinFlip/coin_flip.json"),
            "prompt_generator": generate_coin_flip_prompt,
            "contrastive_prompt_generator": generate_coin_flip_constrastive_prompt,
        },
        "object_tracking": {
            "raw_data": load_object_tracking_data(
                "datasets/BigBench_Object_Tracking/task.json"),
            "prompt_generator": generate_object_tracking_prompt,
            "contrastive_prompt_generator": generate_object_tracking_constrastive_prompt,
        },
        "last_letter": {
            "raw_data": load_last_letter_data(
                "datasets/last_letter/last_letters.json"),
            "prompt_generator": generate_last_letter_prompt,
            "contrastive_prompt_generator":
                generate_last_letter_constrastive_prompt,
        },
        "bigbench_date": {
            "raw_data": load_bigbench_date_data(
                "datasets/bigbench_date/task.json"),
            "prompt_generator": generate_bigbench_date_prompt,
            "contrastive_prompt_generator":
                generate_bigbench_date_contrastive_prompt,
        }
    }

    hf_token = os.environ.get("HF_TOKEN")
    login(token=hf_token)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda"
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # create train val test splits
    if not os.path.exists("data"):
        os.makedirs("data")
    for task in args.tasks:
        if task in datasets:
            train_set, other = train_test_split(
                datasets[task]["raw_data"],
                test_size=0.3,
                random_state=42
            )
            val_set, test_set = train_test_split(
                other,
                test_size=1/3,
                random_state=42
            )
            datasets[task]["train"] = train_set
            datasets[task]["val"] = val_set
            datasets[task]["test"] = test_set
            with open(f"data/{task}_train.json", 'w') as json_file:
                json.dump(train_set, json_file, indent=4)
            with open(f"data/{task}_val.json", 'w') as json_file:
                json.dump(val_set, json_file, indent=4)
            with open(f"data/{task}_test.json", 'w') as json_file:
                json.dump(test_set, json_file, indent=4)

    # run instruct model on training and validation datasets with CP
    # to get training and validation data
    for task in args.tasks:
        if task in datasets:
            print(f"Running inference on {task} train and val datasets.")
            inference(
                datasets[task]["train"],
                datasets[task]["contrastive_prompt_generator"],
                pipeline,
                terminators,
                f"{task}_train_output.json"
            )
            inference(
                datasets[task]["val"],
                datasets[task]["contrastive_prompt_generator"],
                pipeline,
                terminators,
                f"{task}_val_output.json"
            )
        else:
            print(f"Task '{task}' is not recognized. Skipping.")

    # run instruct model on test datasets to get baseline accuracy
    for task in args.tasks:
        if task in datasets:
            print(f"Running inference on {task} test dataset.")
            inference(
                datasets[task]["test"],
                datasets[task]["prompt_generator"],
                pipeline,
                terminators,
                f"{task}_test_output.json"
            )
        else:
            print(f"Task '{task}' is not recognized. Skipping.")
