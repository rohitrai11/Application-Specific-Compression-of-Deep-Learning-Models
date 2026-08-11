#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,time,torch
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]; TASK=Path(__file__).resolve().parent; sys.path[:0]=[str(REPO),str(TASK)]
from common.asc_utils import load_similarity,prune_bert_encoder,save_json,select_layers_notebook_logic,threshold_tag
from PI.common.data import read_dataset
from PI.common.engine import train_one_epoch,evaluate
from PI.common.modeling import load_float_checkpoint,save_float_checkpoint
from PI.common.utils import set_seed,select_device
THESIS_EXPECTED={'90':[1,3,5],'85':[1,2,3,5,7,9],'80':[1,2,3,4,7,9,11]}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--base_model_path',default='PI/checkpoints/bert_base_finetuned'); p.add_argument('--similarity_path',default='PI/similarity/pi_similarity_matrix.pt'); p.add_argument('--dataset_dir',default='PI/dataset'); p.add_argument('--threshold',type=float,required=True); p.add_argument('--epsilon',type=float,default=.01); p.add_argument('--output_dir'); p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); tag=threshold_tag(a.threshold); out=a.output_dir or f'PI/checkpoints/asc_th_{tag}'; set_seed(a.seed); device=select_device(a.device); model,tok,base=load_float_checkpoint(a.base_model_path); sel=select_layers_notebook_logic(load_similarity(a.similarity_path),a.threshold,a.epsilon); pruning=prune_bert_encoder(model.backbone,sel['encoder_indices_zero_based']); exp=THESIS_EXPECTED.get(tag); check={'thesis_expected_removed_layers_one_based':exp,'computed_removed_layers_one_based':sel['encoder_layers_one_based'],'matches_thesis_reported_layer_list':None if exp is None else sel['encoder_layers_one_based']==exp}; train=read_dataset(Path(a.dataset_dir)/'train.pkl'); val=read_dataset(Path(a.dataset_dir)/'val.pkl'); tl=DataLoader(train,shuffle=True,batch_size=a.batch_size); vl=DataLoader(val,batch_size=a.batch_size); model.to(device); opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
    for e in range(1,a.epochs+1): tr=train_one_epoch(model,tl,opt,device); dv=evaluate(model,vl,device,'val'); hist.append({'epoch':e,'train_accuracy':tr['accuracy'],'val_accuracy':dv['accuracy']})
    save_float_checkpoint(model,tok,out,{'stage':'ASC_compressed_and_refinetuned','task':'PI','threshold':a.threshold,'epsilon':a.epsilon,'selection':sel,'pruning':pruning,'reference_check':check,'base_checkpoint_metadata':base,'epochs_after_compression':a.epochs,'refinetuning_seconds':time.perf_counter()-start}); save_json(hist,Path(out)/'training_history.json')
if __name__=='__main__': main()
