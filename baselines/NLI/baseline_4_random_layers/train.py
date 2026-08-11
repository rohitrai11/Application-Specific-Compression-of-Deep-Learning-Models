#!/usr/bin/env python3
from pathlib import Path
import argparse,random,sys,time,torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import load_snli
from common.engine import train_one_epoch,evaluate
from common.modeling import load_float_checkpoint,save_float_checkpoint
from common.utils import set_seed,select_device,save_json
def remove_random_bert_layers(model,n,seed):
 layers=model.backbone.encoder.layer; rng=random.Random(seed); removed=sorted(rng.sample(range(len(layers)),n)); retained=[i for i in range(len(layers)) if i not in removed]; model.backbone.encoder.layer=nn.ModuleList([layers[i] for i in retained]); model.backbone.config.num_hidden_layers=len(retained); return removed,retained
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline1_bert_base'); p.add_argument('--output_dir',default='checkpoints/baseline4_random_6_layers'); p.add_argument('--dataset_dir',default='snli_1.0'); p.add_argument('--remove_layers',type=int,default=6); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--max_length',type=int,default=512); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); set_seed(a.seed); d=select_device(a.device); m,t,source=load_float_checkpoint(a.input_model_path); removed,retained=remove_random_bert_layers(m,a.remove_layers,a.seed); tr=load_snli(Path(a.dataset_dir)/'snli_1.0_train.txt',t,a.max_length); dv=load_snli(Path(a.dataset_dir)/'snli_1.0_dev.txt',t,a.max_length); tl=DataLoader(tr,shuffle=True,batch_size=a.batch_size); dl=DataLoader(dv,batch_size=a.batch_size); m.to(d); opt=torch.optim.Adam(m.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
 for e in range(1,a.epochs+1): x=train_one_epoch(m,tl,opt,d); y=evaluate(m,dl,d,'dev'); hist.append({'epoch':e,'train_accuracy':x['accuracy'],'dev_accuracy':y['accuracy']})
 save_float_checkpoint(m,t,a.output_dir,{'baseline':'Baseline-4','source_checkpoint':a.input_model_path,'source_metadata':source,'number_removed':a.remove_layers,'removed_original_layer_indices_zero_based':removed,'retained_original_layer_indices_zero_based':retained,'seed':a.seed,'post_removal_finetune_epochs':a.epochs,'training_seconds':time.perf_counter()-start}); save_json(hist,Path(a.output_dir)/'training_history.json')
if __name__=='__main__': main()
