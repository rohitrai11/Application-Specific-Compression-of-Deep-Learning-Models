#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,torch,torch.nn as nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.modeling import load_float_checkpoint
from common.utils import directory_size_bytes,save_json
def dynamic_quantize(m):
 try:q=torch.ao.quantization.quantize_dynamic
 except AttributeError:q=torch.quantization.quantize_dynamic
 return q(m.cpu().eval(),{nn.Linear},dtype=torch.qint8)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline1_bert_base'); p.add_argument('--output_dir',default='checkpoints/baseline2_quantized_bert'); a=p.parse_args(); m,t,meta=load_float_checkpoint(a.input_model_path); q=dynamic_quantize(m); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); m.backbone.config.save_pretrained(out); t.save_pretrained(out); torch.save(q.state_dict(),out/'quantized_state_dict.pt'); save_json({'baseline':'Baseline-2 8-bit Quantized BERT','method':'PyTorch dynamic INT8 quantization of nn.Linear modules','source_metadata':meta,'inference_device':'cpu'},out/'metadata.json')
if __name__=='__main__': main()
