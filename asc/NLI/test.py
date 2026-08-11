#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]; TASK=Path(__file__).resolve().parent; sys.path[:0]=[str(REPO),str(TASK)]
from NLI.common.data import load_snli
from NLI.common.engine import evaluate,save_predictions
from NLI.common.modeling import load_float_checkpoint
from NLI.common.utils import select_device,save_json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_model_path',required=True); p.add_argument('--dataset_dir',default='NLI/snli_1.0'); p.add_argument('--output_dir',required=True); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--max_length',type=int,default=512); p.add_argument('--device',default='auto'); a=p.parse_args()
    device=select_device(a.device); model,tok,meta=load_float_checkpoint(a.input_model_path); model.to(device); data=load_snli(Path(a.dataset_dir)/'snli_1.0_test.txt',tok,a.max_length); result=evaluate(model,DataLoader(data,batch_size=a.batch_size),device,'test'); save_predictions(result,a.output_dir,'snli_asc'); metrics={k:v for k,v in result.items() if k not in {'predictions','probabilities'}}; metrics['checkpoint_metadata']=meta; save_json(metrics,Path(a.output_dir)/'metrics.json'); print(f"NLI ASC accuracy: {result['accuracy']:.6f}")
if __name__=='__main__': main()
