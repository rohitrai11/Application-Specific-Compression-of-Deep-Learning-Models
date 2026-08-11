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

from EQA.common.qa_runner import evaluate_model, load_squad_examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_model_path", required=True)
    p.add_argument("--predict_file", default="EQA/squad_dataset/dev-v2.0.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--max_query_length", type=int, default=64)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    model = AutoModelForQuestionAnswering.from_pretrained(args.input_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path, use_fast=True)
    examples = load_squad_examples(args.predict_file, answerable_only=True)

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        eval_examples=examples,
        output_dir=args.output_dir,
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
