#!/bin/bash

BATCH_SIZE=64

MODEL=$1
if [[ ${MODEL} == "" ]]
then
    MODEL="llama-3-8b"
fi
GROUP=$2
if [[ ${GROUP} == "" ]]
then
    GROUP=4
fi
CUDA=$3
if [[ ${CUDA} == "" ]]
then
    CUDA=0
fi

if [[ ${MODEL} == "llama-3-8b" ]]; then
    MODEL_HF="meta-llama/Meta-Llama-3-8B"
    LR=2e-5
elif [[ ${MODEL} == "llama-2-7b" ]]; then
    MODEL_HF="meta-llama/Llama-2-7b-hf"
    LR=8e-5
elif [[ ${MODEL} == "llama-7b" ]]; then
    MODEL_HF="huggyllama/llama-7b"
    LR=8e-5
elif [[ ${MODEL} == "llama-13b" ]]; then
    MODEL_HF="huggyllama/llama-13b"
    LR=8e-5
fi

OUTPUT=./outs/commonsense/${MODEL}_gpr${GROUP}
mkdir -p $OUTPUT

CUDA_VISIBLE_DEVICES=${CUDA} python ./train/finetune.py \
    --model_name_or_path ${MODEL_HF} \
    --sketched_model_path ./models/${MODEL}-int4-gpr${GROUP}.pkl \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${BATCH_SIZE} \
    --max_seq_len 2048 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --num_train_epochs 5 \
    --dtype bf16 \
    --lr_scheduler_type linear \
    --num_warmup_steps 100 \
    --seed 42 \
    --instruction_type single \
    --val_set_size 120 \
    --eval_step 10 \
    --load_last_model \
    --data_path ./ft-training_set/commonsense_170k.json \
    --output_dir $OUTPUT >> ${OUTPUT}/training.log

# Use the following line to enable gradient checkpointing for GPU memory savings:
# --gradient_checkpointing \
