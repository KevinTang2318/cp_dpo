# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import transformers
import json
import os
from threading import Thread
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
    BitsAndBytesConfig,
)

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.2,
        context_len=2048,
        stop_str="</s>",
):
    """Generate answer from prompt with GPT and stream the output"""
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    return generated_text


@torch.inference_mode()
def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        prompt_template,
        system_prompt,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
):
    """Generate answer from prompt with GPT, batch mode"""
    generated_texts = []
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )
    prompts = [prompt_template.get_prompt(messages=[[s, '']], system_prompt=system_prompt) for s in sentences]
    # print(prompts)
    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(device)
    outputs = model.generate(input_ids=input_ids, **generation_kwargs)

    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='auto', type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="llama", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument('--system_prompt', default="", type=str)
    parser.add_argument("--repetition_penalty", type=float, default=1.3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="A file that contains instructions (one instruction per line)")
    parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (default multi-turn)")
    parser.add_argument('--single_tune', action='store_true', help='Whether to use single-tune model')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    prompt_template = "llama"
    system_prompt = "Assistant is a large language model."
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    base_model = model_class.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)

    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
    model.eval()
    print(tokenizer)
    jsons = []

    
    def generate_aqua_constrastive_prompt(sample):
        prompt = f"""
            {sample}
            A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct letter choice>; Wrong answer: <wrong letter choice>; <your reasoning>
        """

        return prompt

    def generate_object_tracking_constrastive_prompt(sample):
        prompt = f"""
            {sample} 
            A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <correct ball color>; Wrong answer: <correct ball color>; <your reasoning>
        """

        return prompt

    def generate_strategy_qa_constrastive_prompt(sample):
        prompt = f"""
            {sample}  first, then provide your reasoning.
            A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>"""

        return prompt

        
    def generate_coin_flip_constrastive_prompt(sample):
        prompt = f"""
            {sample} first, then provide your reasoning. 
            A: Let's give a correct and a wrong answer. Output your answer in the following format: "Correct answer: <yes or no>; Wrong answer: <yes or no>; <your reasoning>"""
        return prompt
    
    
    
    test_cases = pd.read_json(path_or_buf=("test.jsonl"), lines=True)
    for idx, content in test_cases.iterrows():
        
        
        input_sentence = content.question + " Output your answer in the following format:  Reasoning: <your reasoning>. Correct answer: <your answer>; "

        pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        )
        print(pipeline.tokenizer.eos_token_id)
        terminators = [
            
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            pipeline.tokenizer.convert_tokens_to_ids("<|end_of_text|>")
        ]

        prompt = input_sentence
        messages=[
            {"role": "system", "content": "Assistant is a large language model."},
            {"role": "user", "content": prompt}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=512,
            eos_token_id=terminators,
            temperature=0.01,
        )        
        result = (outputs[0]["generated_text"][-1]["content"])


        with open("v12_perform_pipeline_full_test3.json", "a") as json_two:
            json.dump([{"dataset": content.dataset, "ground_truth": content.ground_truth,"question": input_sentence, "answer": result}],json_two,indent = 4)
  


if __name__ == '__main__':
    main()
