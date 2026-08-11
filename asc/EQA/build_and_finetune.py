#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import json

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TASK))

from common.asc_utils import (
    load_similarity,
    prune_bert_encoder,
    save_json,
    select_layers_notebook_logic,
    threshold_tag,
)
from EQA.common.qa_runner import (
    evaluate_model,
    load_squad_examples,
    set_seed,
    train_model,
)

THESIS_EXPECTED = {
    "90": [2, 3, 5, 8],
    "85": [1, 2, 4, 6, 8],
    "80": [1, 2, 3, 4, 6, 8, 10],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", default="EQA/checkpoints/bert_base_finetuned")
    p.add_argument("--similarity_path", default="EQA/similarity/eqa_similarity_matrix.pt")
    p.add_argument("--train_file", default="EQA/squad_dataset/train-v2.0.json")
    p.add_argument("--predict_file", default="EQA/squad_dataset/dev-v2.0.json")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--max_query_length", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    tag = threshold_tag(args.threshold)
    output_dir = args.output_dir or f"EQA/checkpoints/asc_th_{tag}"

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model_path)

    matrix = load_similarity(args.similarity_path)
    selection = select_layers_notebook_logic(matrix, args.threshold, args.epsilon)
    pruning = prune_bert_encoder(
        model.bert, selection["encoder_indices_zero_based"]
    )
    model.config.num_hidden_layers = pruning["num_hidden_layers_after_compression"]

    expected = THESIS_EXPECTED.get(tag)
    check = {
        "thesis_expected_removed_layers_one_based": expected,
        "computed_removed_layers_one_based": selection["encoder_layers_one_based"],
        "matches_thesis_reported_layer_list": (
            None if expected is None else selection["encoder_layers_one_based"] == expected
        ),
    }

    print("ASC selection:", selection)
    print("Pruning:", pruning)
    print("Reference check:", check)

    train_examples = load_squad_examples(args.train_file, answerable_only=True)
    train_info = train_model(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        seed=args.seed,
    )

    metadata = {
        "stage": "ASC_compressed_and_refinetuned",
        "task": "EQA",
        "threshold": args.threshold,
        "epsilon": args.epsilon,
        "selection": selection,
        "pruning": pruning,
        "reference_check": check,
        "epochs_after_compression": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "train_info": train_info,
    }
    save_json(metadata, Path(output_dir) / "asc_metadata.json")

    eval_examples = load_squad_examples(args.predict_file, answerable_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        n_best_size=20,
        max_answer_length=30,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
