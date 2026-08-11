#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_snli
from common.engine import evaluate, save_predictions
from common.modeling import load_float_checkpoint
from common.utils import select_device, save_json


def main():
    parser = argparse.ArgumentParser(description="Test the Pruned BERT NLI baseline.")
    parser.add_argument("--input_model_path", default="checkpoints/baseline2_pruned_bert")
    parser.add_argument("--dataset_dir", default="snli_1.0")
    parser.add_argument("--output_dir", default="predictions/baseline2_pruned_bert")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = select_device(args.device)
    model, tokenizer, metadata = load_float_checkpoint(args.input_model_path)
    model.to(device)
    test_data = load_snli(Path(args.dataset_dir) / "snli_1.0_test.txt", tokenizer, args.max_length)
    result = evaluate(model, DataLoader(test_data, shuffle=False, batch_size=args.batch_size), device, desc="test")
    save_predictions(result, args.output_dir, "snli_pruned_bert")
    metrics = {k: v for k, v in result.items() if k not in {"predictions", "probabilities"}}
    metrics["checkpoint_metadata"] = metadata
    save_json(metrics, Path(args.output_dir) / "metrics.json")
    print(f"SNLI test accuracy: {result['accuracy']:.6f}")
    print(f"Testing time: {result['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
