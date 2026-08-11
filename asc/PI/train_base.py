#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(TASK))

from PI.common.data import read_dataset
from PI.common.engine import train_one_epoch, evaluate
from PI.common.modeling import PIClassifier, save_float_checkpoint
from PI.common.utils import set_seed, select_device, save_json

MODEL_ID = "google-bert/bert-base-uncased"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="PI/dataset")
    p.add_argument("--output_dir", default="PI/checkpoints/bert_base_finetuned")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train_data = read_dataset(Path(args.dataset_dir) / "train.pkl")
    val_data = read_dataset(Path(args.dataset_dir) / "val.pkl")
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)
    model = PIClassifier.from_pretrained_model(MODEL_ID).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        dv = evaluate(model, val_loader, device, desc="val")
        row = {"epoch": epoch,"train_loss_sum": tr["loss_sum"],"train_accuracy": tr["accuracy"],"train_sample_accuracy": tr["sample_accuracy"],"val_loss_sum": dv["loss_sum"],"val_accuracy": dv["accuracy"],"val_sample_accuracy": dv["sample_accuracy"]}
        history.append(row); print(row)
    elapsed = time.perf_counter() - start
    save_float_checkpoint(model, tokenizer, args.output_dir, metadata={"stage":"fine_tuned_uncompressed_model","task":"PI","model_id":MODEL_ID,"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"seed":args.seed,"training_seconds":elapsed})
    save_json(history, Path(args.output_dir) / "training_history.json")

if __name__ == "__main__": main()
