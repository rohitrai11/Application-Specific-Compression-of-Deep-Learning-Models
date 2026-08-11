# Paraphrase Identification (QQP) — Baselines 1–4

PI settings: QQP; labels 0/1; 10 epochs; batch size 32; learning rate 1e-5; accuracy; CLS hidden state -> Linear(hidden_size,2). Put original `train.pkl`, `val.pkl`, and `test.pkl` in `dataset/`; `data_loader.py` is retained for pickle compatibility.

Baseline-1 uses `google-bert/bert-base-uncased`. Baseline-2 Pruned BERT directly uses `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa` without a second pruning step. Baseline-2 quantization dynamically quantizes fine-tuned BERT-base Linear layers to INT8. Baseline-3 uses Distil-BERT, BERT-Medium, BERT-Mini and BERT-Tiny. Baseline-4 removes six encoder blocks reproducibly and stores the layer indices.

Run everything with `bash run_all_baselines.sh`.
