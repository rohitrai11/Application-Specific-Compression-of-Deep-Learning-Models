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

from EQA.common.qa_runner import (
    load_squad_examples,
    set_seed,
    train_model,
    evaluate_model,
)

MODEL_ID = "google-bert/bert-base-uncased"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="EQA/squad_dataset/train-v2.0.json")
    p.add_argument("--predict_file", default="EQA/squad_dataset/dev-v2.0.json")
    p.add_argument("--output_dir", default="EQA/checkpoints/bert_base_finetuned")
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

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)

    train_examples = load_squad_examples(args.train_file, answerable_only=True)
    info = train_model(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        output_dir=args.output_dir,
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

    eval_examples = load_squad_examples(args.predict_file, answerable_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir, use_fast=True)
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_examples=eval_examples,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        n_best_size=20,
        max_answer_length=30,
    )
    print(json.dumps({"train_info": info, "eval_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
