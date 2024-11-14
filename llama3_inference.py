from huggingface_hub import login
import transformers
import torch

from data_loader import load_aqua_data, load_strategy_qa_data


def generate_aqua_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.
        A: Let's give a correct and a wrong answer.
    """

    return prompt


def generate_strategy_qa_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer.
    """

    return prompt


if __name__ == "__main__":

    aqua_dataset = load_aqua_data("datasets/AQuA/test.json")
    strategy_qa_dataset = load_strategy_qa_data("datasets/StrategyQA/task.json")

    login(token="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")

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        num_return_sequences=1,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    eos_token_id = tokenizer.eos_token_id

    print(generate_aqua_constrastive_prompt(aqua_dataset[0]))

    print(pipeline(
        generate_aqua_constrastive_prompt(aqua_dataset[0]),
        eos_token_id=eos_token_id,
    )
    )
