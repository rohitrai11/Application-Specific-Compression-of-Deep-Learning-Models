#!/usr/bin/env python3
from pathlib import Path
import argparse,random,sys,time,torch,torch.nn as nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import read_dataset
from common.engine import train_one_epoch,evaluate
from common.modeling import load_float_checkpoint,save_float_checkpoint
from common.utils import set_seed,select_device,save_json
def remove_layers(model,n,seed,explicit=None):
 layers=model.backbone.encoder.layer; removed=sorted(explicit if explicit is not None else random.Random(seed).sample(range(len(layers)),n)); retained=[i for i in range(len(layers)) if i not in removed]; model.backbone.encoder.layer=nn.ModuleList([layers[i] for i in retained]); model.backbone.config.num_hidden_layers=len(retained); return removed,retained
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline1_bert_base'); p.add_argument('--train_path',default='dataset/train.pkl'); p.add_argument('--val_path',default='dataset/val.pkl'); p.add_argument('--output_dir',default='checkpoints/baseline4_random_6_layers'); p.add_argument('--remove_layers',type=int,default=6); p.add_argument('--removed_layers'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); set_seed(a.seed); d=select_device(a.device); m,t,src=load_float_checkpoint(a.input_model_path); exp=sorted({int(x) for x in a.removed_layers.split(',')}) if a.removed_layers else None; removed,retained=remove_layers(m,a.remove_layers,a.seed,exp); tl=DataLoader(read_dataset(a.train_path),shuffle=True,batch_size=a.batch_size); vl=DataLoader(read_dataset(a.val_path),shuffle=True,batch_size=a.batch_size); m.to(d); opt=torch.optim.Adam(m.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
 for e in range(1,a.epochs+1): tr=train_one_epoch(m,tl,opt,d); va=evaluate(m,vl,d,'validation'); hist.append({'epoch':e,'train_accuracy':tr['accuracy'],'validation_accuracy':va['accuracy']})
 save_float_checkpoint(m,t,a.output_dir,{'baseline':'Baseline-4','source_metadata':src,'removed_original_layer_indices_zero_based':removed,'retained_original_layer_indices_zero_based':retained,'selection':'explicit' if exp else 'random','seed':a.seed,'training_seconds':time.perf_counter()-start}); save_json(hist,Path(a.output_dir)/'training_history.json')
if __name__=='__main__': main()
