import json
import tqdm


def generate_aqua_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.Output your answer in the following format: Answer: <letter choice>; <your reasoning>
    """

    return prompt


def generate_aqua_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Choose one correct answer from the following  options: {sample["options"]}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct letter choice>; Wrong answer: <wrong letter choice>; <your reasoning>
    """

    return prompt


def generate_object_tracking_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.Output your answer in the following format: Answer: <your choice>; <your reasoning>
    """

    return prompt


def generate_object_tracking_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <your choice>; Wrong answer: <your choice>; <your reasoning>
    """

    return prompt


def generate_strategy_qa_prompt(sample):
    prompt = f"""
        Q: {sample["input"]}; Output your answer in the following format: Answer: <YES or NO>; <your reasoning>
    """

    return prompt


def generate_strategy_qa_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>
    """

    return prompt

    
def generate_coin_flip_prompt(sample):
    prompt = f"""
        Q: {sample["question"]}; Output your answer in the following format: 
        Answer: <YES or NO>; <your reasoning>
    """

    return prompt

    
def generate_coin_flip_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]} Answer this question with either "Yes." or "No." first, then provide your reasoning.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>
    """

    return prompt


def generate_last_letter_prompt(sample):
    prompt = f"""
        Q: {sample["question"]}; Output your answer in the following format: 
        Answer: <your answer>; <your reasoning>
    """

    return prompt


def generate_last_letter_constrastive_prompt(sample):
    prompt = f"""
        Q: {sample["question"]}
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <your answer>; Wrong answer: <your answer>; <your reasoning>
    """

    return prompt


def generate_bigbench_date_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.Output your answer in the following format: Answer: <your choice>; <your reasoning>
    """

    return prompt


def generate_bigbench_date_contrastive_prompt(sample):
    prompt = f"""
        Q: {sample["input"]} Choose one correct answer from the following  options: {",".join(list(sample["target_scores"].keys()))}.
        A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <your choice>; Wrong answer: <your choice>; <your reasoning>
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
