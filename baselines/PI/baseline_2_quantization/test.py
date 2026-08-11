#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch,torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig,AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import read_dataset
from common.engine import evaluate,save_predictions
from common.modeling import PIClassifier
from common.utils import save_json
def dq(m):
 try:q=torch.ao.quantization.quantize_dynamic
 except AttributeError:q=torch.quantization.quantize_dynamic
 return q(m.cpu().eval(),{nn.Linear},dtype=torch.qint8)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline2_quantized_bert'); p.add_argument('--test_path',default='dataset/test.pkl'); p.add_argument('--output_dir',default='predictions/baseline2_quantized_bert'); p.add_argument('--batch_size',type=int,default=32); a=p.parse_args(); ck=Path(a.input_model_path); m=dq(PIClassifier(AutoModel.from_config(AutoConfig.from_pretrained(ck)))); m.load_state_dict(torch.load(ck/'quantized_state_dict.pt',map_location='cpu',weights_only=False)); t=AutoTokenizer.from_pretrained(ck); r=evaluate(m,DataLoader(read_dataset(a.test_path),batch_size=a.batch_size),torch.device('cpu'),'test'); save_predictions(r,a.output_dir,'qqp_quantized_bert'); x={k:v for k,v in r.items() if k not in {'predictions','probabilities'}}; save_json(x,Path(a.output_dir)/'metrics.json'); print(f"QQP test accuracy: {r['accuracy']:.6f}")
if __name__=='__main__': main()
