import argparse
import json
import transformers
import torch
import tqdm

from huggingface_hub import login

from data_loader import (
    load_aqua_data, 
    load_strategy_qa_data, 
    load_coin_flip_data, 
    load_object_tracking_data
)


def generate_aqua_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.Output your answer in the following format: Answer: <letter choice>; <your reasoning>
    """

    return prompt

def generate_object_tracking_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.Output your answer in the following format: Answer: <letter choice>; <your reasoning>
    """

    return prompt

def generate_strategy_qa_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
    """

    return prompt

    
def generate_coin_flip_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
    """

    return prompt


def inference(dataset, prompt_generator, pipeline, terminators, out_file_name):
    all_output = []
    for sample in tqdm.tqdm(dataset):
        prompt = prompt_generator(sample)

        messages=[
            # consistent with the CP paper
            {
                "role": "system", 
                "content": "Assistant is a large language model."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            eos_token_id=terminators,
            temperature=0.01,
        )

        sample["llama_output"] = outputs[0]["generated_text"][-1]["content"]
        all_output.append(sample)

    with open(f"llm_output/{out_file_name}", 'w') as json_file:
        json.dump(all_output, json_file, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference on selected tasks.")
    parser.add_argument(
        "tasks",
        nargs="+",
        choices=["aqua", "strategy_qa", "coin_flip", "object_tracking"],
        help="Specify one or more tasks to run inference on."
    )
    args = parser.parse_args()

    datasets = {
        "aqua": (load_aqua_data("datasets/AQuA/test.json"), generate_aqua_prompt, "aqua_output.json"),
        "strategy_qa": (load_strategy_qa_data("datasets/StrategyQA/task.json"), generate_strategy_qa_prompt, "strategy_qa_output.json"),
        "coin_flip": (load_coin_flip_data("datasets/CoinFlip/coin_flip.json"), generate_coin_flip_prompt, "coin_flip_output.json"),
        "object_tracking": (load_object_tracking_data("datasets/BigBench_Object_Tracking/task.json"), generate_object_tracking_prompt, "object_tracking_output.json"),
    }

    login(token="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")

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

    for task in args.tasks:
        if task in datasets:
            dataset, prompt_generator, output_file = datasets[task]
            print(f"Running inference on {task} dataset.")
            inference(
                dataset,
                prompt_generator,
                pipeline,
                terminators,
                output_file
            )
        else:
            print(f"Task '{task}' is not recognized. Skipping.")
