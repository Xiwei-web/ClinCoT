# ClinCoT: Clinical-Aware Visual Chain-of-Thought for Medical Vision Language Models

ClinCoT is a modular research codebase for clinical multimodal reasoning with visual CoT and preference optimization.

## 💡 Overview

<div align=left>
<img src=assets/overview4.png width=90% />
</div>

## 📖  Project Structure
- `clincot/models`: vision/language/projector/checkpoint core
- `clincot/data`: datasets, preprocess, collators, pair builder
- `clincot/methods`: DPO/sDPO loss, visual CoT, trainer core
- `clincot/engine`: distributed, optimizer, scheduler, launcher
- `clincot/inference`: CoT/VQA/report inferencers
- `clincot/evaluation`: VQA/report/factuality metrics
- `clincot/utils`: seed/logging/io/env/registry
- `configs`: training/inference/model/data/experiment configs
- `scripts`: runnable shell scripts

## 📦 Requirements
1. Installation

```Shell
conda create -n clincot python=3.10 -y
conda activate clincot
pip install -r requirements.txt
```

2. Download the required model checkpoints [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.


3. For all the medical datasets, you need firstly apply for the right of access and then download the dataset.

- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)
- [VQA-RAD](https://osf.io/89kps/)
- [SLAKE](https://www.med-vqa.com/slake/)

4. Data Format
See `docs/data_format.md`.

## 🏋️ Train
Single GPU:
```bash
python clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode sft --output_dir runs/sft
```

Multi GPU:
```bash
torchrun --nnodes=1 --nproc_per_node=2 clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode pref --output_dir runs/pref
```

Resume:
```bash
torchrun --nnodes=1 --nproc_per_node=2 clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode pref --output_dir runs/pref --resume
```

## 🏋️ Inference
```bash
python clincot/cli/infer.py \
  --config configs/exp/clincot_base.yaml \
  --checkpoint runs/pref/checkpoint-00001000 \
  --image /path/to/image.png \
  --question "What is the diagnosis?"
```

## 🏋️ Evaluation
```bash
python -m clincot.evaluation.runner --pred outputs/predictions.jsonl --task all
```

