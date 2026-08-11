#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import load_snli
from common.engine import evaluate,save_predictions
from common.modeling import load_float_checkpoint
from common.utils import select_device,save_json
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline1_bert_base'); p.add_argument('--dataset_dir',default='snli_1.0'); p.add_argument('--output_dir',default='predictions/baseline1_bert_base'); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--max_length',type=int,default=512); p.add_argument('--device',default='auto'); a=p.parse_args(); d=select_device(a.device); m,t,meta=load_float_checkpoint(a.input_model_path); m.to(d); r=evaluate(m,DataLoader(load_snli(Path(a.dataset_dir)/'snli_1.0_test.txt',t,a.max_length),batch_size=a.batch_size),d,'test'); save_predictions(r,a.output_dir,'snli_bert_base'); x={k:v for k,v in r.items() if k not in {'predictions','probabilities'}}; x['checkpoint_metadata']=meta; save_json(x,Path(a.output_dir)/'metrics.json'); print(f"SNLI test accuracy: {r['accuracy']:.6f}")
if __name__=='__main__': main()
