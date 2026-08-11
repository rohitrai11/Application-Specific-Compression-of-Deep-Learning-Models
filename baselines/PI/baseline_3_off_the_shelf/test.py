#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import read_dataset
from common.engine import evaluate,save_predictions
from common.modeling import load_float_checkpoint
from common.utils import select_device,save_json
KEYS=['distilbert','bert-medium','bert-mini','bert-tiny']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model_key',required=True,choices=KEYS); p.add_argument('--input_root',default='checkpoints/baseline3'); p.add_argument('--test_path',default='dataset/test.pkl'); p.add_argument('--output_root',default='predictions/baseline3'); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--device',default='auto'); a=p.parse_args(); ck=Path(a.input_root)/a.model_key; out=Path(a.output_root)/a.model_key; d=select_device(a.device); m,t,meta=load_float_checkpoint(ck); m.to(d); r=evaluate(m,DataLoader(read_dataset(a.test_path),batch_size=a.batch_size),d,'test'); save_predictions(r,out,f'qqp_{a.model_key}'); x={k:v for k,v in r.items() if k not in {'predictions','probabilities'}}; x['checkpoint_metadata']=meta; save_json(x,out/'metrics.json'); print(f"QQP accuracy: {r['accuracy']:.6f}")
if __name__=='__main__': main()
