#!/usr/bin/env python3
"""Baseline-2(a): fine-tune Intel's published 90% unstructured sparse BERT on SNLI.

The sparse checkpoint is used directly. No new local pruning operation is
performed in this baseline.
"""
from pathlib import Path
import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_snli
from common.engine import train_one_epoch, evaluate
from common.modeling import NLIClassifier, save_float_checkpoint
from common.utils import set_seed, select_device, save_json

MODEL_ID = "Intel/bert-base-uncased-sparse-90-unstructured-pruneofa"


def main():
    parser = argparse.ArgumentParser(description="Baseline-2(a): Pruned BERT on SNLI.")
    parser.add_argument("--dataset_dir", default="snli_1.0")
    parser.add_argument("--output_dir", default="checkpoints/baseline2_pruned_bert")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_data = load_snli(Path(args.dataset_dir) / "snli_1.0_train.txt", tokenizer, args.max_length)
    dev_data = load_snli(Path(args.dataset_dir) / "snli_1.0_dev.txt", tokenizer, args.max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)

    model = NLIClassifier.from_pretrained_model(MODEL_ID).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, device)
        dv = evaluate(model, dev_loader, device, desc="dev")
        row = {
            "epoch": epoch,
            "train_loss_sum": tr["loss_sum"],
            "train_accuracy": tr["accuracy"],
            "dev_loss_sum": dv["loss_sum"],
            "dev_accuracy": dv["accuracy"],
        }
        history.append(row)
        print(row)

    elapsed = time.perf_counter() - start
    save_float_checkpoint(
        model,
        tokenizer,
        args.output_dir,
        metadata={
            "task": "Natural Language Inference",
            "dataset": "SNLI",
            "baseline": "Baseline-2 Pruned BERT",
            "huggingface_model_id": MODEL_ID,
            "note": "Published sparse checkpoint used directly; no local pruning operation applied.",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
            "training_seconds": elapsed,
        },
    )
    save_json(history, Path(args.output_dir) / "training_history.json")
    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
