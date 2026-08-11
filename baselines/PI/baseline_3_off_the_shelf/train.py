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
MODELS={'distilbert':('Distil-BERT','distilbert/distilbert-base-uncased'),'bert-medium':('BERT-Medium','google/bert_uncased_L-8_H-512_A-8'),'bert-mini':('BERT-Mini','google/bert_uncased_L-4_H-256_A-4'),'bert-tiny':('BERT-Tiny','google/bert_uncased_L-2_H-128_A-2')}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model_key',required=True,choices=MODELS); p.add_argument('--train_path',default='dataset/train.pkl'); p.add_argument('--val_path',default='dataset/val.pkl'); p.add_argument('--output_root',default='checkpoints/baseline3'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); name,mid=MODELS[a.model_key]; out=Path(a.output_root)/a.model_key; set_seed(a.seed); d=select_device(a.device); tok=AutoTokenizer.from_pretrained(mid); tl=DataLoader(read_dataset(a.train_path),shuffle=True,batch_size=a.batch_size); vl=DataLoader(read_dataset(a.val_path),shuffle=True,batch_size=a.batch_size); m=PIClassifier.from_pretrained_model(mid).to(d); opt=torch.optim.Adam(m.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
 for e in range(1,a.epochs+1): tr=train_one_epoch(m,tl,opt,d); va=evaluate(m,vl,d,'validation'); hist.append({'epoch':e,'train_accuracy':tr['accuracy'],'validation_accuracy':va['accuracy']})
 save_float_checkpoint(m,tok,out,{'baseline':'Baseline-3','model_name':name,'huggingface_model_id':mid,'epochs':a.epochs,'seed':a.seed,'training_seconds':time.perf_counter()-start}); save_json(hist,out/'training_history.json')
if __name__=='__main__': main()
