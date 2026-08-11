#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,time,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import read_dataset
from common.engine import train_one_epoch,evaluate
from common.modeling import PIClassifier,save_float_checkpoint
from common.utils import set_seed,select_device,save_json
MODEL_ID='Intel/bert-base-uncased-sparse-90-unstructured-pruneofa'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train_path',default='dataset/train.pkl'); p.add_argument('--val_path',default='dataset/val.pkl'); p.add_argument('--output_dir',default='checkpoints/baseline2_pruned_bert'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); set_seed(a.seed); d=select_device(a.device); tok=AutoTokenizer.from_pretrained(MODEL_ID); tl=DataLoader(read_dataset(a.train_path),shuffle=True,batch_size=a.batch_size); vl=DataLoader(read_dataset(a.val_path),shuffle=True,batch_size=a.batch_size); m=PIClassifier.from_pretrained_model(MODEL_ID).to(d); opt=torch.optim.Adam(m.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
 for e in range(1,a.epochs+1): tr=train_one_epoch(m,tl,opt,d); va=evaluate(m,vl,d,'validation'); hist.append({'epoch':e,'train_accuracy':tr['accuracy'],'validation_accuracy':va['accuracy']})
 save_float_checkpoint(m,tok,a.output_dir,{'task':'Paraphrase Identification','dataset':'QQP','baseline':'Baseline-2 Pruned BERT','huggingface_model_id':MODEL_ID,'note':'Published sparse checkpoint used directly; no local pruning operation applied.','epochs':a.epochs,'seed':a.seed,'training_seconds':time.perf_counter()-start}); save_json(hist,Path(a.output_dir)/'training_history.json')
if __name__=='__main__': main()
