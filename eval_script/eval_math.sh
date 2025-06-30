#!/bin/bash

EPOCH=3

MODEL=$1
if [[ ${MODEL} == "" ]]
then
    MODEL=llama-3-8b
fi
GROUP=$2
if [[ ${GROUP} == "" ]]
then
    GROUP=1
fi
ADAPTER_PATH=$3
if [[ ${ADAPTER_PATH} == "" ]]
then
    ADAPTER_PATH=./ckpts/math/${MODEL}_gpr${GROUP}/epoch_${EPOCH}
fi
CUDA=$4
if [[ ${CUDA} == "" ]]
then
    CUDA=0
fi
BATCH_SIZE=$5
if [[ ${BATCH_SIZE} == "" ]]
then
    BATCH_SIZE=8
fi

if [[ ${MODEL} == "llama-3-8b" ]]; then
    MODEL_HF="meta-llama/Meta-Llama-3-8B"
elif [[ ${MODEL} == "llama-2-7b" ]]; then
    MODEL_HF="meta-llama/Llama-2-7b-hf"
elif [[ ${MODEL} == "llama-7b" ]]; then
    MODEL_HF="huggyllama/llama-7b"
elif [[ ${MODEL} == "llama-13b" ]]; then
    MODEL_HF="huggyllama/llama-13b"
fi

datasets=(AQuA MultiArith AddSub SingleEq SVAMP mawps gsm8k)

for dataset in "${datasets[@]}"
do
    OUTPUT=${ADAPTER_PATH}/$dataset
    mkdir -p $OUTPUT

    CUDA_VISIBLE_DEVICES=${CUDA} python ./eval/run_math_parallel.py \
        --data_path ./dataset/$dataset/test.json \
        --model_name_or_path ${MODEL_HF} \
        --sketched_model_path ./models/${MODEL}-int4-gpr${GROUP}.pkl \
        --adapter_path ${ADAPTER_PATH} \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --seed 1234 \
        --dtype bf16 \
        --dataset ${dataset} \
        --output_dir ${OUTPUT} >> ${OUTPUT}/eval.log
done
