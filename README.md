# Application Specific Compression of Deep Learning Models

GitHub-ready implementation of the BERT experiments for the thesis chapter
**Application Specific Compression of Deep Learning Models**.

This repository contains:

1. all four baseline categories for:
   - Natural Language Inference (NLI),
   - Paraphrase Identification (PI),
   - Extractive Question Answering (EQA);
2. the Application Specific Compression (ASC) implementation for all three tasks;
3. task-specific training, evaluation, data-loading and execution scripts;
4. the threshold-based layer-removal logic derived from the supplied ASC notebook;
5. a machine-readable file map and thesis-reference results.

---

## Repository layout

```text
Application_Specific_Compression_GitHub/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── FILE_MAP.txt
│   └── thesis_reference.json
│
├── scripts/
│   ├── run_all_baselines.sh
│   ├── run_all_asc.sh
│   └── run_everything.sh
│
├── baselines/
│   │
│   ├── NLI/
│   │   ├── common/
│   │   │   ├── data.py
│   │   │   ├── engine.py
│   │   │   ├── modeling.py
│   │   │   └── utils.py
│   │   ├── baseline_1_bert_base/
│   │   │   ├── train.py
│   │   │   ├── test.py
│   │   │   └── run.sh
│   │   ├── baseline_2_pruned_bert/
│   │   │   ├── train.py
│   │   │   ├── test.py
│   │   │   └── run.sh
│   │   ├── baseline_2_quantization/
│   │   │   ├── quantize.py
│   │   │   ├── test.py
│   │   │   └── run.sh
│   │   ├── baseline_3_off_the_shelf/
│   │   │   ├── train.py
│   │   │   ├── test.py
│   │   │   └── run_all.sh
│   │   ├── baseline_4_random_layers/
│   │   │   ├── train.py
│   │   │   ├── test.py
│   │   │   └── run.sh
│   │   ├── snli_1.0/
│   │   ├── checkpoints/
│   │   ├── predictions/
│   │   ├── README.md
│   │   └── requirements.txt
│   │
│   ├── PI/
│   │   ├── common/
│   │   │   ├── data.py
│   │   │   ├── engine.py
│   │   │   ├── modeling.py
│   │   │   └── utils.py
│   │   ├── baseline_1_bert_base/
│   │   ├── baseline_2_pruned_bert/
│   │   ├── baseline_2_quantization/
│   │   ├── baseline_3_off_the_shelf/
│   │   ├── baseline_4_random_layers/
│   │   ├── data_loader.py
│   │   ├── dataset/
│   │   ├── checkpoints/
│   │   ├── predictions/
│   │   ├── README.md
│   │   └── requirements.txt
│   │
│   └── EQA/
│       ├── common/
│       │   ├── __init__.py
│       │   └── qa_runner.py
│       ├── baseline_1_bert_base/
│       │   ├── train_eval.sh
│       │   └── eval.sh
│       ├── baseline_2_pruned_bert/
│       │   ├── train_eval.sh
│       │   └── eval.sh
│       ├── baseline_2_quantization/
│       │   ├── quantize_and_eval.py
│       │   └── run.sh
│       ├── baseline_3_off_the_shelf/
│       │   ├── run_all.sh
│       │   ├── train_eval_distilbert.sh
│       │   ├── train_eval_bert_medium.sh
│       │   ├── train_eval_bert_mini.sh
│       │   └── train_eval_bert_tiny.sh
│       ├── baseline_4_random_layers/
│       │   ├── train_eval.sh
│       │   ├── eval.sh
│       │   └── train_eval_explicit_layers_example.sh
│       ├── squad_dataset/
│       ├── checkpoints/
│       ├── results/
│       ├── logs/
│       ├── README.md
│       └── requirements.txt
│
└── asc/
    ├── common/
    │   ├── __init__.py
    │   └── asc_utils.py
    │
    ├── NLI/
    │   ├── common/
    │   ├── train_base.py
    │   ├── compute_similarity.py
    │   ├── build_and_finetune.py
    │   ├── test.py
    │   ├── run_all.sh
    │   ├── snli_1.0/
    │   ├── checkpoints/
    │   ├── similarity/
    │   └── predictions/
    │
    ├── PI/
    │   ├── common/
    │   ├── data_loader.py
    │   ├── train_base.py
    │   ├── compute_similarity.py
    │   ├── build_and_finetune.py
    │   ├── test.py
    │   ├── run_all.sh
    │   ├── dataset/
    │   ├── checkpoints/
    │   ├── similarity/
    │   └── predictions/
    │
    ├── EQA/
    │   ├── common/
    │   ├── train_base.py
    │   ├── compute_similarity.py
    │   ├── build_and_finetune.py
    │   ├── test.py
    │   ├── run_all.sh
    │   ├── squad_dataset/
    │   ├── checkpoints/
    │   ├── similarity/
    │   └── predictions/
    │
    ├── run_all_tasks.sh
    ├── run_from_existing_base_models.sh
    ├── thesis_reference.json
    ├── README.md
    └── requirements.txt
```

The exact complete file list is also stored in `docs/FILE_MAP.txt`.

---

# 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/rohitrai11/Application-Specific-Compression-of-Deep-Learning-Models.git
cd Application-Specific-Compression-of-Deep-Learning-Models

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for the full experiments.

---

# 2. Required datasets

Dataset files are intentionally not included in the repository.

## NLI: SNLI

For the baseline experiments place:

```text
baselines/NLI/snli_1.0/
├── snli_1.0_train.txt
├── snli_1.0_dev.txt
└── snli_1.0_test.txt
```

For ASC place the same files in:

```text
asc/NLI/snli_1.0/
├── snli_1.0_train.txt
├── snli_1.0_dev.txt
└── snli_1.0_test.txt
```

## PI: QQP

The reference PI implementation uses the already prepared pickle splits:

```text
train.pkl
val.pkl
test.pkl
```

For baselines:

```text
baselines/PI/dataset/
├── train.pkl
├── val.pkl
└── test.pkl
```

For ASC:

```text
asc/PI/dataset/
├── train.pkl
├── val.pkl
└── test.pkl
```

The repository keeps `data_loader.py` because these serialized dataset objects
may require that module while unpickling.

## EQA: SQuAD2.0

Place:

```text
train-v2.0.json
dev-v2.0.json
```

under:

```text
baselines/EQA/squad_dataset/
```

and:

```text
asc/EQA/squad_dataset/
```

The EQA experiment uses the answerable portion of SQuAD2.0.

---

# 3. Baseline models

The thesis uses four baseline categories.

## Baseline-1

Full BERT-base:

```text
google-bert/bert-base-uncased
```

## Baseline-2

### Pruned BERT

```text
Intel/bert-base-uncased-sparse-90-unstructured-pruneofa
```

This is the pretrained 90% unstructured sparse model used in the experiments.
The repository does not apply an additional pruning operation.

### 8-bit Quantized BERT

The fine-tuned BERT-base task model is converted to a dynamic INT8 model.

## Baseline-3

Off-the-shelf compressed BERT variants:

```text
Distil-BERT:
distilbert/distilbert-base-uncased

BERT-Medium:
google/bert_uncased_L-8_H-512_A-8

BERT-Mini:
google/bert_uncased_L-4_H-256_A-4

BERT-Tiny:
google/bert_uncased_L-2_H-128_A-2
```

## Baseline-4

BERT-base with six randomly removed encoder layers.

The original thesis/reference files do not provide the exact six random layers.
The repository therefore records the seed and actual selected layer indices for
reproducibility rather than claiming an unknown historical selection.

---

# 4. Run baseline experiments

Run each task from its own baseline directory so all relative paths match the
original/reference code layout.

## NLI baselines

```bash
cd baselines/NLI

bash baseline_1_bert_base/run.sh
bash baseline_2_pruned_bert/run.sh
bash baseline_2_quantization/run.sh
bash baseline_3_off_the_shelf/run_all.sh
bash baseline_4_random_layers/run.sh
```

Or:

```bash
bash run_all_baselines.sh
```

## PI baselines

```bash
cd baselines/PI

bash baseline_1_bert_base/run.sh
bash baseline_2_pruned_bert/run.sh
bash baseline_2_quantization/run.sh
bash baseline_3_off_the_shelf/run_all.sh
bash baseline_4_random_layers/run.sh
```

Or:

```bash
bash run_all_baselines.sh
```

## EQA baselines

```bash
cd baselines/EQA

bash baseline_1_bert_base/train_eval.sh
bash baseline_2_pruned_bert/train_eval.sh
bash baseline_2_quantization/run.sh
bash baseline_3_off_the_shelf/run_all.sh
bash baseline_4_random_layers/train_eval.sh
```

Or:

```bash
bash run_all_baselines.sh
```

From the repository root you can execute all baseline tasks:

```bash
bash scripts/run_all_baselines.sh
```

---

# 5. ASC method

ASC follows the task-specific compression workflow:

```text
BERT-base
    |
    v
fine-tune on target task
    |
    v
compute layer-output similarity using task training data
    |
    v
13 x 13 cosine-similarity matrix
(embedding + 12 BERT encoder outputs)
    |
    v
select redundant layers using threshold
    |
    v
physically remove redundant encoder blocks
    |
    v
fine-tune compressed model
    |
    v
evaluate compressed ASC model
```

ASC thresholds used in the experiments are:

```text
0.90
0.85
0.80
```

The notebook used an offset:

```python
effective_threshold = threshold - 0.01
```

which is exposed by the ASC scripts as:

```text
--epsilon 0.01
```

---

# 6. Run ASC for NLI

From repository root:

```bash
cd asc
```

First train the uncompressed task-specific model:

```bash
python3 NLI/train_base.py
```

Compute the application-specific similarity matrix:

```bash
python3 NLI/compute_similarity.py
```

Build and re-fine-tune the three ASC models:

```bash
python3 NLI/build_and_finetune.py --threshold 0.90
python3 NLI/build_and_finetune.py --threshold 0.85
python3 NLI/build_and_finetune.py --threshold 0.80
```

Test an ASC checkpoint:

```bash
python3 NLI/test.py \
  --input_model_path NLI/checkpoints/asc_th_90 \
  --output_dir NLI/predictions/asc_th_90
```

Or run the complete pipeline:

```bash
bash NLI/run_all.sh
```

---

# 7. Run ASC for PI

From `asc/`:

```bash
python3 PI/train_base.py
python3 PI/compute_similarity.py

python3 PI/build_and_finetune.py --threshold 0.90
python3 PI/build_and_finetune.py --threshold 0.85
python3 PI/build_and_finetune.py --threshold 0.80
```

Test:

```bash
python3 PI/test.py \
  --input_model_path PI/checkpoints/asc_th_90 \
  --output_dir PI/predictions/asc_th_90
```

Or:

```bash
bash PI/run_all.sh
```

---

# 8. Run ASC for EQA

From `asc/`:

```bash
python3 EQA/train_base.py
python3 EQA/compute_similarity.py

python3 EQA/build_and_finetune.py --threshold 0.90
python3 EQA/build_and_finetune.py --threshold 0.85
python3 EQA/build_and_finetune.py --threshold 0.80
```

Evaluate:

```bash
python3 EQA/test.py \
  --input_model_path EQA/checkpoints/asc_th_90 \
  --output_dir EQA/predictions/asc_th_90
```

Or:

```bash
bash EQA/run_all.sh
```

---

# 9. Run all ASC tasks

From the repository root:

```bash
bash scripts/run_all_asc.sh
```

Equivalent:

```bash
cd asc
bash run_all_tasks.sh
```

If the full BERT-base models have already been fine-tuned and are stored at:

```text
asc/NLI/checkpoints/bert_base_finetuned/
asc/PI/checkpoints/bert_base_finetuned/
asc/EQA/checkpoints/bert_base_finetuned/
```

skip the first fine-tuning stage with:

```bash
cd asc
bash run_from_existing_base_models.sh
```

---

# 10. Run every experiment

From the repository root:

```bash
bash scripts/run_everything.sh
```

This runs all baselines first and then all ASC experiments.

This is computationally expensive.

---

# 11. Similarity-matrix output

ASC saves each task's dataset-averaged similarity matrix in four formats.

Example:

```text
asc/NLI/similarity/
├── nli_similarity_matrix.pt
├── nli_similarity_matrix.npy
├── nli_similarity_matrix.csv
└── nli_similarity_matrix.json
```

The matrix has 13 representation stages:

```text
embed, enc1, enc2, ..., enc12
```

The CSV format can be directly used to generate heatmaps.

For a quick software smoke test only:

```bash
cd asc
python3 NLI/compute_similarity.py --max_samples 100
```

Do not use a sample-limited matrix for the final reported experiment.

---

# 12. ASC layer-selection reference

Reported one-based encoder layers removed in the thesis are:

| Task | 90% | 85% | 80% |
|---|---|---|---|
| NLI | 1, 3, 5, 8 | 1, 2, 4, 5, 7 | 1, 2, 3, 4, 6, 8, 10 |
| PI | 1, 3, 5 | 1, 2, 3, 5, 7, 9 | 1, 2, 3, 4, 7, 9, 11 |
| EQA | 2, 3, 5, 8 | 1, 2, 4, 6, 8 | 1, 2, 3, 4, 6, 8, 10 |

The code does not use these lists to perform compression. It calculates the
layers from the task-specific similarity matrix and records whether the computed
selection agrees with the thesis list.

---

# 13. Task-specific training settings

## NLI

```text
Dataset: SNLI
Epochs: 5
Batch size: 32
Learning rate: 1e-5
Maximum sequence length: 512
Metric: Accuracy
```

## PI

```text
Dataset: QQP
Epochs: 10
Batch size: 32
Learning rate: 1e-5
Metric: Accuracy
```

## EQA

```text
Dataset: SQuAD2.0 answerable questions
Epochs: 3
Batch size: 32
Learning rate: 3e-5
Warmup ratio: 0.1
Maximum sequence length: 384
Document stride: 128
Maximum query length: 64
Metrics: Exact Match and F1
```

---

# 14. Generated artifacts

Training output should not be committed to GitHub.

The `.gitignore` files exclude:

```text
checkpoints/
predictions/
similarity/
results/
logs/
dataset files
Python cache files
```

Only source code, configuration, documentation and empty placeholder directories
should be committed.

---

# 15. Reproducibility notes

- The baseline and ASC implementations use the task-specific settings from the
  supplied reference files.
- Pruned BERT directly uses
  `Intel/bert-base-uncased-sparse-90-unstructured-pruneofa`.
- The ASC similarity pass uses the application training data.
- The dataset-averaged similarity matrix is used for layer selection.
- Padding tokens are excluded from the similarity calculation.
- ASC layer selections, retained layers, thresholds and compression metadata are
  saved with each compressed model.
- The exact Baseline-4 historical random layer indices are not present in the
  supplied source material, so the implementation records the actual random
  selection used by each reproducible run.

---

# 16. Citation

If you use this repository, cite the publication associated with the ASC method:

**Application Specific Compression of Deep Learning Models**, presented at
ACM CoDS-COMAD, December 2024.

Add the final bibliographic metadata/DOI in this section before public release
if desired.
