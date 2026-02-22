# Experiment Protocol

## Goal
Standardize ClinCoT training/evaluation for reproducibility.

## Procedure
1. Freeze dataset split files and record checksums.
2. Choose one config under `configs/exp/`.
3. Run training with fixed seed.
4. Save checkpoints and training logs.
5. Run inference on frozen test split.
6. Run evaluation and store metrics JSON.

## Logging Requirements
- Commit hash
- Config path
- Seed
- GPU count and device type
- Final checkpoint path
- Metrics output
