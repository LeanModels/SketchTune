# Code Repository for [ICML 2025] *Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation*

**SketchTune** is a novel framework that significantly simplifies and accelerates the adaptation of large pre-trained language models (LLMs). Unlike traditional parameter-efficient fine-tuning (PEFT) methods which typically add low rank adapters atop frozen weights, SketchTune compresses the base model itself through learned *sketching*. This produces a smaller, fully trainable model that retains high performance without relying on low rank constraints.

By integrating compression and adaptation into a single framework, SketchTune avoids the inefficiencies of dual path computations found in other PEFT techniques. It delivers faster training and inference, reduced memory usage, and empirically often surpasses leading methods like LoRA, DoRA, S2FT, and LoftQ. It achieves up to **3× smaller base models** and **7× fewer trainable parameters** than baselines, while boosting accuracy on benchmarks like GSM8K by over **14 %**.

For more details, please refer to our [research paper](https://arxiv.org/abs/2410.06364).

> We are committed to reproducibility and have released all [compressed base models](https://huggingface.co/LeanQuant/SketchTune) and [fine-tuned checkpoints](https://huggingface.co/LeanQuant/SketchTune-ckpts) to facilitate easy replication of our results. If you encounter any issues, please feel free to open an issue on GitHub.

---

## Environment Setup

Clone the repository and navigate to it:

```
git clone https://github.com/LeanModels/SketchTune
cd SketchTune
```

(Optional) Set up a conda environment:

```
conda create -n sketchtune python=3.10
conda activate sketchtune
```

Install Python dependencies:

```
pip install -r requirements.txt
```

---

## Download Compressed Base Models

Get access to the Llama models on Hugging Face. Llama 2 and 3 are gated and require manual access approval:

* [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* [https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

Download all sketched (compressed) base models for fine-tuning to the `models` directory:

```
huggingface-cli download LeanQuant/SketchTune --local-dir ./models
```

To download a specific model, visit the model list here:
[https://huggingface.co/LeanQuant/SketchTune/tree/main](https://huggingface.co/LeanQuant/SketchTune/tree/main)

---

## Fine-Tuning

To fine-tune on commonsense datasets:

```
bash train_script/finetune_commonsense.sh llama-3-8b 4 0
```

To fine-tune on math datasets:

```
bash train_script/finetune_math.sh llama-3-8b 1 0
```

**Arguments:**

* Model name: `llama-3-8b`, `llama-2-7b`, `llama-7b`, or `llama-13b`
* (Optional) Groups per row (GPR): 1, 2, 4, or 8
* (Optional) CUDA device index

**Notes:**

* Outputs are saved in `outs/commonsense/llama-3-8b_gpr4` or a similar directory.
* Model checkpoints are automatically saved as `sketched_params.pkl` at the end of each epoch.
* Training logs and loss curves are sent to Weights & Biases. Run `wandb login` to enable logging.

**Requirements:**

* A CUDA-compatible GPU with at least 24GB of VRAM.

---

## Evaluation

To evaluate on commonsense datasets:

```
bash eval_script/eval_commonsense.sh llama-3-8b 4 ./outs/commonsense/llama-3-8b_gpr4/epoch_1 0 8
```

To evaluate on math datasets:

```
bash eval_script/eval_math.sh llama-3-8b 4 ./outs/math/llama-3-8b_gpr4/epoch_1 0 8
```

**Arguments:**

* Model name: `llama-3-8b`, `llama-2-7b`, `llama-7b`, or `llama-13b`
* (Optional) GPR: 1, 2, 4, or 8
* (Optional) Path of the directory containing the fine-tuned checkpoint (`sketched_params.pkl`)
* (Optional) CUDA device index
* (Optional) Batch size

Results are saved to the same directory as the checkpoint. Each dataset result is written to its own subfolder.

***We have released all fine-tuned checkpoints for reproducibility. To download all fine-tuned checkpoints to the `ckpts` directory:***
```
huggingface-cli download LeanQuant/SketchTune-ckpts --local-dir ./ckpts
```

## Results

Evaluation results of SketchTune and baselines on **commonsense** benchmarks:

| Model         | Method           | Base Model (GB)  | Trainable Param (M)  | BoolQ  | PIQA | SIQA | HellaSwag  | Wino | ARC-e  | ARC-c  | OBQA | Average |
|---------------|------------------|------------------|----------------------|--------|------|------|------------|------|--------|--------|------|---------|
| Llama-7B      | Full FT          | 13.48            | 6,738.4              | 70.3   | 84.2 | 80.1 | 92.3       | 85.4 | 86.6   | 72.8   | 83.4 | 81.9    |
|               | LoRA             | 13.48            | 55.9                 | 69.2   | 81.7 | 78.4 | 83.4       | 80.8 | 79.0   | 62.4   | 78.4 | 76.7    |
|               | DoRA             | 13.48            | 56.6                 | 68.5   | 82.9 | 79.6 | 84.8       | 80.8 | 81.4   | 65.8   | 81.0 | 78.1    |
|               | S2FT             | 13.48            | 54.6                 | 72.7   | 83.7 | 79.6 | 93.4       | 83.5 | 86.1   | 72.2   | 83.4 | 81.8    |
|               | SketchTune,GPR=4 | 4.02             | 87.0                 | 72.1   | 85.6 | 80.2 | 93.7       | 84.6 | 86.2   | 71.0   | 84.8 | 82.3    |
| Llama-13B     | Full FT          | 26.03            | 13,015.9             | 74.5   | 86.3 | 81.3 | 94.4       | 86.9 | 89.7   | 77.9   | 88.8 | 85.0    |
|               | LoRA             | 26.03            | 87.2                 | 72.1   | 83.5 | 80.5 | 90.5       | 83.7 | 82.8   | 68.3   | 82.4 | 80.5    |
|               | DoRA             | 26.03            | 88.5                 | 72.4   | 84.9 | 81.5 | 92.4       | 84.2 | 84.2   | 69.6   | 82.8 | 81.5    |
|               | S2FT             | 26.03            | 84.6                 | 74.2   | 85.7 | 80.7 | 94.9       | 86.4 | 88.4   | 76.3   | 87.8 | 84.3    |
|               | SketchTune,GPR=4 | 7.36             | 136.3                | 73.9   | 87.4 | 82.5 | 95.6       | 86.1 | 90.3   | 75.7   | 89.4 | 85.1    |
| Llama-2-7B    | Full FT          | 13.48            | 6,738.4              | 74.7   | 84.9 | 78.7 | 93.7       | 84.1 | 87.5   | 75.2   | 85.0 | 83.0    |
|               | LoRA             | 13.48            | 55.9                 | 69.8   | 79.9 | 79.5 | 83.6       | 82.6 | 79.8   | 64.7   | 81.0 | 77.6    |
|               | DoRA             | 13.48            | 56.6                 | 71.8   | 83.7 | 76.0 | 89.1       | 82.6 | 83.7   | 68.2   | 82.4 | 79.7    |
|               | S2FT             | 13.48            | 54.6                 | 72.9   | 86.1 | 80.2 | 94.3       | 85.5 | 87.2   | 74.6   | 83.4 | 83.0    |
|               | SketchTune,GPR=4 | 4.05             | 87.0                 | 73.3   | 86.2 | 81.2 | 94.1       | 85.4 | 87.6   | 75.2   | 85.8 | 83.6    |
| Llama-3-8B    | Full FT          | 16.06            | 8,030.3              | 73.9   | 86.2 | 79.1 | 93.1       | 85.8 | 88.1   | 78.2   | 84.0 | 83.6    |
|               | LoRA             | 16.06            | 56.2                 | 70.8   | 85.2 | 79.7 | 92.5       | 84.9 | 88.9   | 78.7   | 84.4 | 82.5    |
|               | DoRA             | 16.06            | 57.0                 | 74.6   | 89.3 | 79.9 | 95.5       | 85.6 | 90.5   | 80.4   | 85.8 | 85.2    |
|               | S2FT             | 16.06            | 56.2                 | 75.0   | 89.0 | 80.7 | 96.5       | 88.0 | 92.5   | 83.4   | 87.8 | 86.6    |
|               | SketchTune,GPR=4 | 5.92             | 88.1                 | 75.0   | 90.2 | 82.7 | 95.9       | 88.2 | 92.6   | 82.1   | 89.4 | 87.0    |

Evaluation results on **math** benchmarks:

| Model           | Method            | Base Model (GB) | Trainable Param (M) | MultiArith | GSM8K | AddSub | AQuA | SingleEq | SVAMP | MAWPS | Average|
|-----------------|-------------------|-----------------|---------------------|------------|-------|--------|------|----------|-------|-------|--------|
| LLaMA-7B        | Full FT           | 13.48           | 6,738.4             | 98.8       | 43.1  | 91.1   | 20.9 | 94.3     | 60.6  | 88.2  | 71.0   |
|                 | LoRA              | 13.48           | 55.9                | 98.0       | 40.0  | 91.2   | 21.7 | 93.1     | 56.7  | 85.3  | 69.7   |
|                 | DoRA              | 13.48           | 56.6                | 97.3       | 38.9  | 89.6   | 22.4 | 93.9     | 58.4  | 85.3  | 69.4   |
|                 | S2FT              | 13.48           | 54.6                | 98.8       | 41.3  | 91.4   | 21.3 | 93.5     | 58.4  | 86.1  | 70.1   |
|                 | SketchTune,GPR=1  | 3.89            | 21.8                | 97.8       | 36.5  | 89.9   | 25.2 | 90.7     | 55.7  | 86.6  | 68.9   |
|                 | SketchTune,GPR=2  | 3.93            | 43.5                | 96.8       | 39.0  | 92.2   | 20.1 | 92.7     | 55.5  | 86.6  | 69.0   |
|                 | SketchTune,GPR=4  | 4.02            | 87.0                | 98.3       | 39.7  | 90.9   | 22.0 | 93.5     | 58.0  | 87.4  | 70.0   |
|                 | SketchTune,GPR=8  | 4.19            | 174.1               | 98.3       | 40.6  | 91.9   | 19.7 | 95.1     | 57.5  | 88.7  | 70.3   |
| LLaMA-13B       | Full FT           | 26.03           | 13,015.9            | 98.3       | 47.6  | 92.9   | 26.0 | 95.1     | 65.7  | 88.7  | 73.5   |
|                 | LoRA              | 26.03           | 87.2                | 97.5       | 47.8  | 89.9   | 20.5 | 94.3     | 61.2  | 87.4  | 71.2   |
|                 | DoRA              | 26.03           | 88.5                | 97.2       | 48.1  | 90.6   | 20.9 | 93.9     | 63.8  | 88.2  | 71.8   |
|                 | S2FT              | 26.03           | 84.6                | 97.7       | 48.4  | 90.4   | 22.8 | 95.5     | 63.9  | 87.8  | 72.4   |
|                 | SketchTune,GPR=1  | 7.14            | 34.1                | 97.2       | 44.0  | 88.6   | 26.0 | 91.7     | 64.9  | 85.7  | 71.2   |
|                 | SketchTune,GPR=2  | 7.21            | 68.2                | 98.2       | 46.9  | 91.1   | 27.2 | 93.9     | 61.8  | 86.6  | 72.2   |
|                 | SketchTune,GPR=4  | 7.36            | 136.3               | 98.5       | 47.8  | 91.9   | 24.0 | 95.9     | 64.2  | 89.1  | 73.1   |
|                 | SketchTune,GPR=8  | 7.67            | 272.6               | 98.8       | 47.6  | 92.2   | 25.2 | 95.5     | 66.8  | 87.4  | 73.4   |
| LLaMA2-7B       | Full FT           | 13.48           | 6,738.4             | 99.3       | 47.5  | 91.1   | 24.4 | 96.7     | 62.5  | 89.1  | 72.9   |
|                 | LoRA              | 13.48           | 55.9                | 97.5       | 44.0  | 91.2   | 20.9 | 94.1     | 59.2  | 85.7  | 70.4   |
|                 | DoRA              | 13.48           | 56.6                | 98.2       | 43.8  | 90.1   | 24.4 | 94.5     | 59.1  | 89.1  | 71.3   |
|                 | S2FT              | 13.48           | 54.6                | 98.5       | 44.3  | 91.1   | 25.2 | 94.7     | 61.8  | 88.2  | 72.0   |
|                 | SketchTune,GPR=1  | 3.92            | 21.8                | 98.0       | 41.4  | 89.6   | 26.4 | 92.9     | 59.3  | 89.1  | 71.0   |
|                 | SketchTune,GPR=2  | 3.97            | 43.5                | 98.8       | 43.5  | 92.2   | 20.5 | 95.3     | 59.9  | 89.1  | 71.3   |
|                 | SketchTune,GPR=4  | 4.05            | 87.0                | 99.3       | 46.5  | 91.1   | 23.2 | 94.5     | 59.8  | 88.2  | 71.8   |
|                 | SketchTune,GPR=8  | 4.23            | 174.1               | 98.7       | 46.5  | 93.9   | 24.0 | 96.7     | 61.7  | 90.3  | 73.1   |
| LLaMA3-8B       | Full FT           | 16.06           | 8,030.3             | 99.2       | 62.0  | 93.9   | 26.8 | 96.7     | 74.0  | 91.2  | 77.7   |
|                 | LoRA              | 16.06           | 56.2                | 99.5       | 61.6  | 92.7   | 25.6 | 96.3     | 73.8  | 90.8  | 77.2   |
|                 | DoRA              | 16.06           | 57.0                | 98.8       | 62.7  | 92.2   | 26.8 | 96.9     | 74.0  | 91.2  | 77.5   |
|                 | S2FT              | 16.06           | 56.2                | 99.7       | 65.8  | 93.7   | 31.5 | 97.8     | 76.0  | 92.4  | 79.6   |
|                 | SketchTune,GPR=1  | 5.77            | 22.0                | 97.8       | 66.3  | 90.1   | 26.8 | 95.5     | 79.8  | 90.8  | 78.2   |
|                 | SketchTune,GPR=2  | 5.81            | 44.0                | 98.3       | 69.4  | 90.6   | 29.5 | 94.3     | 76.8  | 91.2  | 78.6   |
|                 | SketchTune,GPR=4  | 5.92            | 88.1                | 99.2       | 68.2  | 91.4   | 30.7 | 97.0     | 76.2  | 92.4  | 79.3   |
|                 | SketchTune,GPR=8  | 6.10            | 176.2               | 99.7       | 68.8  | 92.7   | 29.1 | 98.6     | 77.5  | 92.9  | 79.9   |

## Acknowledgements

This repository is built upon [S2FT](https://github.com/Infini-AI-Lab/S2FT) and [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). We would like to thank the authors of these projects for their outstanding work.

## Citation

If you found our work interesting or useful, please consider citing us.
```
@inproceedings{
    zhang2025sketch,
    title={Sketch to Adapt: Fine-Tunable Sketches for Efficient {LLM} Adaptation},
    author={Tianyi Zhang and Junda Su and Aditya Desai and Oscar Wu and Zhaozhuo Xu and Anshumali Shrivastava},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=zZXOXhxO6I}
}
```