# Natural Language Inference (SNLI) — Baselines 1–4

Settings: SNLI 1.0; contradiction=0, neutral=1, entailment=2; 5 epochs; batch size 32; learning rate 1e-5; max length 512; accuracy; head `CLS -> Linear(hidden_size, 6) -> Linear(6, 3)`.

Place SNLI train/dev/test files under `snli_1.0/`.

Baseline-1 uses `google-bert/bert-base-uncased`.

Baseline-2 Pruned BERT directly fine-tunes `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa`; no additional `torch.nn.utils.prune` operation is applied. Run `bash baseline_2_pruned_bert/run.sh`.

Baseline-2 quantization applies dynamic INT8 to the fine-tuned BERT-base model. Run `bash baseline_2_quantization/run.sh`.

Baseline-3 uses Distil-BERT, BERT-Medium, BERT-Mini and BERT-Tiny. Run `bash baseline_3_off_the_shelf/run_all.sh`.

Baseline-4 removes six encoder layers reproducibly and records the selected indices. Run `bash baseline_4_random_layers/run.sh`.
