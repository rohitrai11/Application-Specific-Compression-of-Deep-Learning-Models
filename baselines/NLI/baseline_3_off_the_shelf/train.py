#!/usr/bin/env python3
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

MODELS = {
    "distilbert": ("Distil-BERT", "distilbert/distilbert-base-uncased"),
    "bert-medium": ("BERT-Medium", "google/bert_uncased_L-8_H-512_A-8"),
    "bert-mini": ("BERT-Mini", "google/bert_uncased_L-4_H-256_A-4"),
    "bert-tiny": ("BERT-Tiny", "google/bert_uncased_L-2_H-128_A-2"),
}


def main():
    p = argparse.ArgumentParser(description="Fine-tune an off-the-shelf compressed BERT-family model on SNLI.")
    p.add_argument("--model_key", required=True, choices=MODELS.keys())
    p.add_argument("--dataset_dir", default="snli_1.0")
    p.add_argument("--output_root", default="checkpoints/baseline3")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    display_name, model_id = MODELS[args.model_key]
    output_dir = Path(args.output_root) / args.model_key
    set_seed(args.seed)
    device = select_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_data = load_snli(Path(args.dataset_dir) / "snli_1.0_train.txt", tokenizer, args.max_length)
    dev_data = load_snli(Path(args.dataset_dir) / "snli_1.0_dev.txt", tokenizer, args.max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)

    model = NLIClassifier.from_pretrained_model(model_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        print(f"{display_name}: epoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, device)
        dv = evaluate(model, dev_loader, device, desc="dev")
        history.append({"epoch": epoch, "train_accuracy": tr["accuracy"], "dev_accuracy": dv["accuracy"]})
        print(history[-1])

    elapsed = time.perf_counter() - start
    save_float_checkpoint(model,tokenizer,output_dir,metadata={"baseline":"Baseline-3","model_name":display_name,"huggingface_model_id":model_id,"epochs":args.epochs,"batch_size":args.batch_size,"learning_rate":args.learning_rate,"seed":args.seed,"training_seconds":elapsed})
    save_json(history, output_dir / "training_history.json")
    print(f"Saved {display_name} to {output_dir}")

if __name__ == "__main__": main()
