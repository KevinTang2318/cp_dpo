from huggingface_hub import login
import json
import transformers
import torch
import tqdm

from data_loader import load_aqua_data, load_strategy_qa_data, load_coin_flip_data, load_object_tracking_data


def generate_aqua_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct letter choice>; Wrong answer: <wrong letter choice>; <your reasoning>
    """

    return prompt

def generate_object_tracking_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct ball color>; Wrong answer: <correct ball color>; <your reasoning>
    """

    return prompt

def generate_strategy_qa_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>
    """

    return prompt

    
def generate_coin_flip_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>
    """

    return prompt

if __name__ == "__main__":

    aqua_dataset = load_aqua_data("datasets/AQuA/test.json")
    strategy_qa_dataset = load_strategy_qa_data("datasets/StrategyQA/task.json")
    coin_flip_dataset = load_coin_flip_data("datasets/CoinFlip/coin_flip.json")
    object_tracking_dataset = load_object_tracking_data("datasets/BigBench_Object_Tracking/task.json")

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

    object_tracking_output = []
    for sample in tqdm.tqdm(object_tracking_dataset):
        prompt = generate_object_tracking_constrastive_prompt(sample)
        print(prompt)
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
        object_tracking_output.append(sample)

    with open("llm_output/object_tracking_output.json", 'w') as json_file:
        json.dump(object_tracking_output, json_file, indent=4)
    # coin_flip_output = []
    # for sample in tqdm.tqdm(coin_flip_dataset):
    #     prompt = generate_coin_flip_constrastive_prompt(sample)

    #     messages=[
    #         {"role": "system", "content": "Assistant is a large language model."},
    #         {"role": "user", "content": prompt}
    #     ]

    #     outputs = pipeline(
    #         messages,
    #         max_new_tokens=1024,
    #         eos_token_id=terminators,
    #         temperature=0.01,
    #     )

    #     sample["llama_output"] = outputs[0]["generated_text"][-1]["content"]
    #     coin_flip_output.append(sample)

    # with open("llm_output/coin_flip_output.json", 'w') as json_file:
    #     json.dump(coin_flip_output, json_file, indent=4)

    # # generate llm output form AQuA dataset:
    # aqua_output = []
    # for sample in tqdm.tqdm(aqua_dataset):
    #     prompt = generate_aqua_constrastive_prompt(sample)

    #     messages=[
    #         {"role": "system", "content": "Assistant is a large language model."},
    #         {"role": "user", "content": prompt}
    #     ]

    #     outputs = pipeline(
    #         messages,
    #         max_new_tokens=1024,
    #         eos_token_id=terminators,
    #         temperature=0.01,
    #     )

    #     sample["llama_output"] = outputs[0]["generated_text"][-1]["content"]
    #     aqua_output.append(sample)

    # with open("llm_output/aqua_output.json", 'w') as json_file:
    #     json.dump(aqua_output, json_file, indent=4)

    # # generate llm output form StrategyQA dataset:
    # strategy_qa_output = []
    # for sample in tqdm.tqdm(strategy_qa_dataset):
    #     prompt = generate_strategy_qa_constrastive_prompt(sample)

    #     messages=[
    #         {"role": "system", "content": "Assistant is a large language model."},
    #         {"role": "user", "content": prompt}
    #     ]

    #     outputs = pipeline(
    #         messages,
    #         max_new_tokens=1024,
    #         eos_token_id=terminators,
    #         temperature=0.01,
    #     )

    #     sample["llama_output"] = outputs[0]["generated_text"][-1]["content"]
    #     strategy_qa_output.append(sample)

    # with open("llm_output/strategy_qa_output.json", 'w') as json_file:
    #     json.dump(strategy_qa_output, json_file, indent=4)