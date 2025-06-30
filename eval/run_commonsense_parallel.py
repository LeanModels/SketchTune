import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import argparse
import re
import json
import pickle
import torch

from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
)
from accelerate import Accelerator
from accelerate.utils import gather_object

from utils.utils import print_rank_0, set_random_seed
from utils.model_utils import load_hf_tokenizer, create_hf_model
from utils.generation_utils import generate_completions

from tailor_utils import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

i_prompt = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
'''  
    
def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    sentence = sentence.lower()
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


@torch.no_grad()
def main(args):
    accelerator = Accelerator()
    set_random_seed(args.seed)
    t_test_data = json.load(open(args.data_path, 'r'))

    prompts = []
    for example in t_test_data:
        prompt = i_prompt.format_map(example)
        prompts.append(prompt)
    print_rank_0(prompts[0])

    print_rank_0("Loading model and tokenizer...")
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.padding_side = "left"
    print_rank_0(f"tokenizer pad side: {tokenizer.padding_side}")

    model = create_hf_model(AutoModelForCausalLM,
                        args.model_name_or_path,
                        tokenizer)
    with open(args.sketched_model_path, 'rb') as file:
        quantizers = pickle.load(file) # Quantized LUT + code.

    with open(os.path.join(args.adapter_path, "sketched_params.pkl"), 'rb') as file:
        quant_grids = pickle.load(file) # LUT trained, finetuned.

    replace_layers(model, quantizers, quant_grids)
    model = model.to(accelerator.device)
    # args.dtype = torch.float16 if args.dtype == 'fp16' else torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    # model = model.to(args.dtype)
    model.eval()
    print_rank_0('model is dtype: {}'.format(model.dtype))
                        
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
    )
    accelerator.wait_for_everyone()
    device = accelerator.device
    with accelerator.split_between_processes(prompts) as prompt:
        model_outputs = []
        outputs = generate_completions(
            model=model,
            device=device,
            tokenizer=tokenizer,
            prompts=prompt,
            max_new_tokens=256,                          
            batch_size=args.per_device_eval_batch_size,
            stop_id_sequences=[[tokenizer.eos_token]],
            verbose=False,
            generation_config = generation_config
        )
        model_outputs.extend(outputs)
    outputs = gather_object(model_outputs)

    save_outputs = []
    correct = 0
    for example, output in zip(t_test_data, outputs):
        example['raw_output'] = output
        target = example["answer"].lower()
        predict = extract_answer(args, output)
        if target == predict:
            correct += 1
        example['prediction'] = predict
        save_outputs.append(example)

    print_rank_0(f"Saving outputs to {args.output_dir}")

    weighted_acc = correct/len(t_test_data)
    print_rank_0("Result {:.1f}, total: {}".format(weighted_acc * 100, len(t_test_data)))


    with open(os.path.join(args.output_dir, f"model_predictions.jsonl"), "w") as fout:
        for example in save_outputs:
            fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="", required=True)
    parser.add_argument("--dataset", type=str, default="", required=True)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--sketched_model_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Inference data type')
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="batch size for evaluation.")
    args = parser.parse_args()

    main(args) 
        
    