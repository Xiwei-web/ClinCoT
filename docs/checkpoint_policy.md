# Checkpoint Policy

- Checkpoint naming: `checkpoint-<global_step>`
- State file: `training_state.pt`
- Latest pointer: `latest.json`
- Retention: keep last K checkpoints (default K=3)

## Resume
Resume always reads from `latest.json`.

## Export
Use `clincot/cli/export.py` to export plain `state_dict` for deployment.
