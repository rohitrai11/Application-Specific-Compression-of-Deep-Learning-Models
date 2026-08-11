#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TASK))

from common.asc_utils import compute_dataset_similarity, save_similarity_bundle
from EQA.common.qa_runner import (
    QATensorDataset,
    build_features,
    load_squad_examples,
)


def prepare_inputs(batch, device):
    inputs = {
        "input_ids": batch["input_ids"].to(device, dtype=torch.long),
        "attention_mask": batch["attention_mask"].to(device, dtype=torch.long),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(device, dtype=torch.long)
    return inputs, inputs["attention_mask"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_model_path", default="EQA/checkpoints/bert_base_finetuned")
    p.add_argument("--train_file", default="EQA/squad_dataset/train-v2.0.json")
    p.add_argument("--output_dir", default="EQA/similarity")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--max_query_length", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional feature limit for smoke tests; omit for full training set.")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    device = torch.device(
        "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.input_model_path)
    model.to(device)
    model.eval()

    examples = load_squad_examples(args.train_file, answerable_only=True)
    features = build_features(
        examples,
        tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False,
    )
    dataset = QATensorDataset(features, is_training=False)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)

    matrix, count = compute_dataset_similarity(
        backbone=model.bert,
        dataloader=loader,
        device=device,
        prepare_inputs=prepare_inputs,
        max_samples=args.max_samples,
    )
    paths = save_similarity_bundle(
        matrix, args.output_dir, "eqa_similarity_matrix", count
    )
    print(f"Processed SQuAD features: {count}")
    print(f"Saved similarity bundle: {paths}")


if __name__ == "__main__":
    main()
