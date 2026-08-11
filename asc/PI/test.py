#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]; TASK=Path(__file__).resolve().parent; sys.path[:0]=[str(REPO),str(TASK)]
from PI.common.data import read_dataset
from PI.common.engine import evaluate,save_predictions
from PI.common.modeling import load_float_checkpoint
from PI.common.utils import select_device,save_json
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_model_path',required=True); p.add_argument('--dataset_dir',default='PI/dataset'); p.add_argument('--output_dir',required=True); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--device',default='auto'); a=p.parse_args(); device=select_device(a.device); model,_,meta=load_float_checkpoint(a.input_model_path); model.to(device); result=evaluate(model,DataLoader(read_dataset(Path(a.dataset_dir)/'test.pkl'),batch_size=a.batch_size),device,'test'); save_predictions(result,a.output_dir,'qqp_asc'); metrics={k:v for k,v in result.items() if k not in {'predictions','probabilities'}}; metrics['checkpoint_metadata']=meta; save_json(metrics,Path(a.output_dir)/'metrics.json'); print(f"PI ASC accuracy: {result['accuracy']:.6f}")
if __name__=='__main__': main()
