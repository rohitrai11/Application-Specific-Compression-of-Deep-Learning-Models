# Application Specific Compression (ASC) — BERT for NLI, PI and EQA

This folder implements the **Application Specific Compression (ASC)** pipeline for NLI, PI and EQA.

ASC workflow:

```text
pretrained BERT-base
        ↓
fine-tune on target task
        ↓
forward pass over target-task training data
        ↓
13 × 13 cosine-similarity matrix (embedding + 12 encoder outputs)
        ↓
threshold-based redundant-layer selection
        ↓
remove selected BERT encoder blocks
        ↓
fine-tune compressed model again
        ↓
evaluate ASC model
```

The similarity matrix is computed once for each task and reused at thresholds `0.90`, `0.85`, and `0.80`. Hidden-state index `0` is the embedding output and hidden-state indices `1..12` correspond to encoder layers `1..12`; encoder index is therefore `hidden_state_index - 1`.

The notebook example compares against `threshold - 0.01`; this implementation exposes that offset as `--epsilon 0.01`. It intentionally uses the dataset-averaged similarity matrix and masks padding tokens.

Reported removed one-based encoder layers:

| Task | 90% | 85% | 80% |
|---|---|---|---|
| NLI | 1, 3, 5, 8 | 1, 2, 4, 5, 7 | 1, 2, 3, 4, 6, 8, 10 |
| PI | 1, 3, 5 | 1, 2, 3, 5, 7, 9 | 1, 2, 3, 4, 7, 9, 11 |
| EQA | 2, 3, 5, 8 | 1, 2, 4, 6, 8 | 1, 2, 3, 4, 6, 8, 10 |

Run all three tasks from this directory:

```bash
bash run_all_tasks.sh
```

Or, if fine-tuned BERT-base checkpoints already exist under each task's `checkpoints/bert_base_finetuned/` directory:

```bash
bash run_from_existing_base_models.sh
```

Each task writes its similarity matrix as `.pt`, `.npy`, `.csv`, and `.json`, plus ASC metadata in the compressed checkpoint.
