from huggingface_hub import login
import json
import transformers
import torch
import tqdm

from data_loader import load_aqua_data, load_strategy_qa_data


def generate_aqua_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct letter choice>; Wrong answer: <wrong letter choice>; <your reasoning>
    """

    return prompt


def generate_strategy_qa_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>
    """

    return prompt


if __name__ == "__main__":

    aqua_dataset = load_aqua_data("datasets/AQuA/test.json")
    strategy_qa_dataset = load_strategy_qa_data("datasets/StrategyQA/task.json")

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

    # generate llm output form AQuA dataset:
    aqua_output = []
    for sample in tqdm.tqdm(aqua_dataset):
        prompt = generate_aqua_constrastive_prompt(sample)

        messages=[
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=1024,
            eos_token_id=terminators,
            temperature=0.01,
        )

        sample["llama_output"] = outputs[0]["generated_text"][-1]["content"]
        aqua_output.append(sample)

    with open("llm_output/aqua_output.json", 'w') as json_file:
        json.dump(aqua_output, json_file, indent=4)
