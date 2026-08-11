#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,time,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import load_snli
from common.engine import train_one_epoch,evaluate
from common.modeling import NLIClassifier,save_float_checkpoint
from common.utils import set_seed,select_device,save_json
MODEL_ID='google-bert/bert-base-uncased'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_dir',default='snli_1.0'); p.add_argument('--output_dir',default='checkpoints/baseline1_bert_base'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--max_length',type=int,default=512); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); set_seed(a.seed); d=select_device(a.device); tok=AutoTokenizer.from_pretrained(MODEL_ID); tr=load_snli(Path(a.dataset_dir)/'snli_1.0_train.txt',tok,a.max_length); dv=load_snli(Path(a.dataset_dir)/'snli_1.0_dev.txt',tok,a.max_length); tl=DataLoader(tr,shuffle=True,batch_size=a.batch_size); dl=DataLoader(dv,batch_size=a.batch_size); m=NLIClassifier.from_pretrained_model(MODEL_ID).to(d); opt=torch.optim.Adam(m.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
 for e in range(1,a.epochs+1):
  x=train_one_epoch(m,tl,opt,d); y=evaluate(m,dl,d,'dev'); hist.append({'epoch':e,'train_accuracy':x['accuracy'],'dev_accuracy':y['accuracy']})
 save_float_checkpoint(m,tok,a.output_dir,{'baseline':'Baseline-1','model_name':'BERT-base','huggingface_model_id':MODEL_ID,'epochs':a.epochs,'batch_size':a.batch_size,'learning_rate':a.learning_rate,'max_length':a.max_length,'seed':a.seed,'training_seconds':time.perf_counter()-start}); save_json(hist,Path(a.output_dir)/'training_history.json')
if __name__=='__main__': main()
