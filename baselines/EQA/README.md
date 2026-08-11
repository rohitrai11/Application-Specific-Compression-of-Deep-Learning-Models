# EQA Baselines — Application Specific Compression (ASC)

Extractive Question Answering baselines for SQuAD2.0. Baseline categories are BERT-base; Pruned and INT8 BERT; Distil-BERT/BERT-Medium/BERT-Mini/BERT-Tiny; and BERT with six randomly removed encoder layers.

Use answerable SQuAD2.0 questions, seed 42, 3 epochs, batch size 32, learning rate 3e-5, warmup ratio 0.1, max sequence length 384, doc stride 128 and max query length 64.

Pruned BERT directly uses `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa`; no additional pruning is applied. Run all with `bash run_all_baselines.sh` after placing `train-v2.0.json` and `dev-v2.0.json` in `squad_dataset/`.
