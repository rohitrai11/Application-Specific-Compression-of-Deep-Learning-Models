#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering,AutoTokenizer
from common.qa_runner import evaluate_model,load_squad_examples,set_seed
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/eqa_bert_base'); p.add_argument('--output_dir',default='results/eqa_quantized_bert'); p.add_argument('--predict_file',default='squad_dataset/dev-v2.0.json'); p.add_argument('--answerable_only',action='store_true'); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); set_seed(a.seed); tok=AutoTokenizer.from_pretrained(a.input_model_path,use_fast=True); dense=AutoModelForQuestionAnswering.from_pretrained(a.input_model_path).cpu().eval()
    try: model=torch.ao.quantization.quantize_dynamic(dense,{nn.Linear},dtype=torch.qint8)
    except AttributeError: model=torch.quantization.quantize_dynamic(dense,{nn.Linear},dtype=torch.qint8)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); tok.save_pretrained(out); dense.config.save_pretrained(out); torch.save(model.state_dict(),out/'quantized_state_dict.pt'); examples=load_squad_examples(a.predict_file,a.answerable_only); metrics=evaluate_model(model,tok,examples,str(out),torch.device('cpu'),a.batch_size,384,128,64,20,30); (out/'quantization_metadata.json').write_text(json.dumps({'source_checkpoint':a.input_model_path,'quantization':'dynamic_int8','evaluation_device':'cpu','metrics':metrics},indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
