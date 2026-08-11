#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader

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
from NLI.common.data import load_snli
from NLI.common.engine import train_one_epoch, evaluate
from NLI.common.modeling import load_float_checkpoint, save_float_checkpoint
from NLI.common.utils import set_seed, select_device

THESIS_EXPECTED = {
    "90": [1, 3, 5, 8],
    "85": [1, 2, 4, 5, 7],
    "80": [1, 2, 3, 4, 6, 8, 10],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", default="NLI/checkpoints/bert_base_finetuned")
    p.add_argument("--similarity_path", default="NLI/similarity/nli_similarity_matrix.pt")
    p.add_argument("--dataset_dir", default="NLI/snli_1.0")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    tag = threshold_tag(args.threshold)
    output_dir = args.output_dir or f"NLI/checkpoints/asc_th_{tag}"

    set_seed(args.seed)
    device = select_device(args.device)

    model, tokenizer, base_metadata = load_float_checkpoint(args.base_model_path)
    matrix = load_similarity(args.similarity_path)

    selection = select_layers_notebook_logic(matrix, args.threshold, args.epsilon)
    pruning = prune_bert_encoder(
        model.backbone, selection["encoder_indices_zero_based"]
    )

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

    train_data = load_snli(Path(args.dataset_dir) / "snli_1.0_train.txt", tokenizer, args.max_length)
    dev_data = load_snli(Path(args.dataset_dir) / "snli_1.0_dev.txt", tokenizer, args.max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    dev_loader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
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
        model, tokenizer, output_dir,
        metadata={
            "stage": "ASC_compressed_and_refinetuned",
            "task": "NLI",
            "threshold": args.threshold,
            "epsilon": args.epsilon,
            "selection": selection,
            "pruning": pruning,
            "reference_check": check,
            "base_checkpoint_metadata": base_metadata,
            "epochs_after_compression": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "refinetuning_seconds": elapsed,
        },
    )
    save_json(history, Path(output_dir) / "training_history.json")
    print(f"Saved ASC NLI model to {output_dir}")


if __name__ == "__main__":
    main()
