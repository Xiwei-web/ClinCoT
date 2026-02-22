# Data Format

## SFT JSONL row
```json
{
  "image": "data/images/example.png",
  "question": "What abnormality is visible?",
  "answer": "Left pleural effusion."
}
```

## Preference JSONL row
```json
{
  "image": "data/images/example.png",
  "question": "What abnormality is visible?",
  "chosen": "Left pleural effusion.",
  "rejected": "No abnormality detected.",
  "score_chosen": 0.9,
  "score_rejected": 0.1,
  "rejected_image": "data/noised/example.png"
}
```

## Candidate JSONL row (for pair building)
```json
{
  "image": "data/images/example.png",
  "question": "What abnormality is visible?",
  "candidates": [
    {"answer": "Left pleural effusion.", "score": 0.9},
    {"answer": "No abnormality detected.", "score": 0.1}
  ]
}
```
