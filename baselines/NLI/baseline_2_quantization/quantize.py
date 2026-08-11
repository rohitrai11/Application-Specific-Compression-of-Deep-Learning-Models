#!/usr/bin/env python3
"""Baseline-2(b): post-training dynamic INT8 quantization of fine-tuned BERT-base."""
from pathlib import Path
import argparse
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.modeling import load_float_checkpoint
from common.utils import directory_size_bytes, save_json


def dynamic_quantize(model):
    model = model.cpu().eval()
    try:
        quantize_dynamic = torch.ao.quantization.quantize_dynamic
    except AttributeError:
        quantize_dynamic = torch.quantization.quantize_dynamic
    return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def main():
    p = argparse.ArgumentParser(description="Create the 8-bit Quantized BERT baseline.")
    p.add_argument("--input_model_path", default="checkpoints/baseline1_bert_base")
    p.add_argument("--output_dir", default="checkpoints/baseline2_quantized_bert")
    args = p.parse_args()

    model, tokenizer, source_metadata = load_float_checkpoint(args.input_model_path)
    quantized = dynamic_quantize(model)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.backbone.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(quantized.state_dict(), out / "quantized_state_dict.pt")
    save_json({"baseline":"Baseline-2 8-bit Quantization","method":"PyTorch dynamic INT8 quantization of nn.Linear modules","source_checkpoint":str(args.input_model_path),"source_metadata":source_metadata,"inference_device":"cpu"}, out / "metadata.json")
    print(f"Saved quantized checkpoint to {out}")
    print(f"Checkpoint directory size: {directory_size_bytes(out) / (1024**2):.2f} MiB")

if __name__ == "__main__": main()
