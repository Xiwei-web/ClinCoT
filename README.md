# ClinCoT

ClinCoT is a modular research codebase for clinical multimodal reasoning with visual CoT and preference optimization.

## 1. Project Structure
- `clincot/models`: vision/language/projector/checkpoint core
- `clincot/data`: datasets, preprocess, collators, pair builder
- `clincot/methods`: DPO/sDPO loss, visual CoT, trainer core
- `clincot/engine`: distributed, optimizer, scheduler, launcher
- `clincot/inference`: CoT/VQA/report inferencers
- `clincot/evaluation`: VQA/report/factuality metrics
- `clincot/utils`: seed/logging/io/env/registry
- `configs`: training/inference/model/data/experiment configs
- `scripts`: runnable shell scripts

## 2. Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 3. Data Format
See `docs/data_format.md`.

## 4. Training
Single GPU:
```bash
python clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode sft --output_dir runs/sft
```

Multi GPU:
```bash
torchrun --nnodes=1 --nproc_per_node=4 clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode pref --output_dir runs/pref
```

Resume:
```bash
torchrun --nnodes=1 --nproc_per_node=4 clincot/cli/train.py --config configs/exp/clincot_base.yaml --mode pref --output_dir runs/pref --resume
```

## 5. Inference
```bash
python clincot/cli/infer.py \
  --config configs/exp/clincot_base.yaml \
  --checkpoint runs/pref/checkpoint-00001000 \
  --image /path/to/image.png \
  --question "What is the diagnosis?"
```

## 6. Evaluation
```bash
python -m clincot.evaluation.runner --pred outputs/predictions.jsonl --task all
```

## 7. Reproducibility
- fixed seed in config
- deterministic options in `clincot/utils/seed.py`
- exact config snapshots under `configs/exp/`
- checkpoint lifecycle in `docs/checkpoint_policy.md`
