import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


import copy
import torch
import json
import random
import math
import time
import argparse
import pickle
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler

from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    int_or_float,
)

from utils.model_utils import (
    load_hf_tokenizer,
    create_hf_model,
    save_hf_format,
    get_optimizer_grouped_parameters,
    make_model_gradient_checkpointing_compatible,
    print_throughput
)

from utils.data_utils import SupervisedDataset, DataCollatorForSupervisedDataset

from tailor_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="SketchTune Training")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["./LLM-Adapters/ft-training_set/commonsense_170k.json"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        '--sketched_model_path',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=0,
        help="Size of the validation set. If 0, no validation set is used.",
    )
    parser.add_argument(
        "--load_last_model", action="store_true", help="only save the last model"
    )
    parser.add_argument(
        "--eval_step",
        type=int,
        default=80,
        help="size of eval_step",
    )
    parser.add_argument(
        "--eval_delay",
        type=int_or_float,
        default=0,
        help="eval after certain steps if it is an integer, or eval after certain ratio of steps if it is a float",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Training data type",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout rate of the model."
    )
    parser.add_argument(
        "--instruction_type", type=str, choices=["single", "multi"], default="single"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    wandb.init(project='SketchTune', name=args.output_dir.split('/')[-1], config=args)

    device = torch.device("cuda:0")

    args.global_rank = 0

    set_random_seed(args.seed)

    ## Load model and tokenizer
    tokenizer = load_hf_tokenizer(
        args.model_name_or_path, fast_tokenizer=True
    )  
    tokenizer.model_max_length = args.max_seq_len
    print_rank_0(f"Tokenizer: {tokenizer.model_max_length}", args.global_rank)

    print_rank_0(f"Loading model from {args.model_name_or_path}", args.global_rank)
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        dropout=args.dropout,
    )
    if isinstance(args.sketched_model_path, str):
        for param in model.parameters():
            param.requires_grad = False
        with open(args.sketched_model_path, 'rb') as file:
            quantizers = pickle.load(file)

        replace_layers(model, quantizers)
        if args.gradient_checkpointing:
            model = make_model_gradient_checkpointing_compatible(model)
        model.save_pretrained = type(model.save_pretrained)(save_pretrained, model)

    ## Load Data
    if len(args.data_path) == 1 and ".json" in args.data_path[0]:
        print_rank_0(f"------json Data: {args.data_path[0]}", args.global_rank)
        train_dataset = SupervisedDataset(
            data_path=args.data_path[0],
            tokenizer=tokenizer,
            instruction_type=args.instruction_type,
            args=args,
        )
        if args.val_set_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset,
                [len(train_dataset) - args.val_set_size, args.val_set_size],
            )
            print_rank_0(
                f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}",
                args.global_rank,
            )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise ValueError(
            "Only json format is supported for now. Please check your data format."
        )

    train_sampler = RandomSampler(train_dataset)
    if args.val_set_size > 0:
        val_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
    )
    if args.val_set_size > 0:
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.learning_rate
    )

    AdamOptimizer = AdamW
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    model = model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ## Training
    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        model.train()
        return perplexity, losses.item()

    print_rank_0("***** Running training *****", args.global_rank)
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    print_rank_0(formatted_args, args.global_rank)

    # print trainable parameters
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Number of trainable parameters: {num}", args.global_rank)

    total_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    current_step_count = 0

    # lr_plot = []
    best_val_loss = float("inf")
    final_saved_model_index = 0
    best_model = None

    args.eval_step = args.eval_step * args.gradient_accumulation_steps

    args.eval_delay = (
        args.eval_delay
        if isinstance(args.eval_delay, int)
        else int(args.eval_delay * total_training_steps)
    )

    wandb.watch(model)
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        model.train()
        mean_loss = 0.0
        step_loss = 0.0
        start = time.time()
        for step, batch in enumerate(train_dataloader):
            # lr_plot.append(lr)
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            total_loss = loss / args.gradient_accumulation_steps
            total_loss.backward()
            
            mean_loss += total_loss.item()
            step_loss += total_loss.item()
            
            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(train_dataloader)):
                lr = lr_scheduler.get_last_lr()[1] \
                    if len(lr_scheduler.get_last_lr()) > 1 \
                    else lr_scheduler.get_last_lr()[0]
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                current_step_count += 1

                end = time.time()
                print_content = f"Loss: {step_loss:.4f}, Mean Loss: {mean_loss/current_step_count:.4f}, LR: {lr:.8f}, Step: {current_step_count}/{total_training_steps}, Step Time: {end - start:.4f} sec"
                metrics = {
                    'loss': step_loss,
                    'lr': lr,
                    'step': current_step_count,
                    'step_time': float(end - start),
                }
                print(print_content)
                print(print_content, file=sys.stderr)
                wandb.log(metrics)
                
                step_loss = 0.0
                
                start = time.time()

            if (
                current_step_count % args.eval_step == 0
                and args.val_set_size > 0
                and not args.load_last_model
                and current_step_count >= args.eval_delay
            ):
                ppl, val_loss = evaluation(model, val_dataloader)
                print_rank_0(
                    f"Validation perplexity: {ppl}, Validation loss: {val_loss}",
                    args.global_rank,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.global_rank == 0:
                        best_model = copy.deepcopy(model).to("cpu")
                    final_saved_model_index = current_step_count

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} Train loss: {mean_loss/len(train_dataloader)}",
            args.global_rank,
        )
        
        model.save_pretrained(os.path.join(args.output_dir, f'epoch_{epoch}'))


if __name__ == "__main__":
    main()
