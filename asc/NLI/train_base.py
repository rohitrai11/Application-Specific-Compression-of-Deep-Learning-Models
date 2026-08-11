#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,time,torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[1]; TASK=Path(__file__).resolve().parent; sys.path[:0]=[str(REPO),str(TASK)]
from NLI.common.data import load_snli
from NLI.common.engine import train_one_epoch,evaluate
from NLI.common.modeling import NLIClassifier,save_float_checkpoint
from NLI.common.utils import set_seed,select_device,save_json
MODEL_ID='google-bert/bert-base-uncased'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset_dir',default='NLI/snli_1.0'); p.add_argument('--output_dir',default='NLI/checkpoints/bert_base_finetuned'); p.add_argument('--epochs',type=int,default=5); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=1e-5); p.add_argument('--max_length',type=int,default=512); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); a=p.parse_args(); set_seed(a.seed); device=select_device(a.device); tok=AutoTokenizer.from_pretrained(MODEL_ID); train=load_snli(Path(a.dataset_dir)/'snli_1.0_train.txt',tok,a.max_length); dev=load_snli(Path(a.dataset_dir)/'snli_1.0_dev.txt',tok,a.max_length); tl=DataLoader(train,shuffle=True,batch_size=a.batch_size); dl=DataLoader(dev,batch_size=a.batch_size); model=NLIClassifier.from_pretrained_model(MODEL_ID).to(device); opt=torch.optim.Adam(model.parameters(),lr=a.learning_rate); hist=[]; start=time.perf_counter()
    for e in range(1,a.epochs+1):
        tr=train_one_epoch(model,tl,opt,device); dv=evaluate(model,dl,device,'dev'); row={'epoch':e,'train_loss_sum':tr['loss_sum'],'train_accuracy':tr['accuracy'],'dev_loss_sum':dv['loss_sum'],'dev_accuracy':dv['accuracy']}; hist.append(row); print(row)
    save_float_checkpoint(model,tok,a.output_dir,{'stage':'fine_tuned_uncompressed_model','task':'NLI','model_id':MODEL_ID,'epochs':a.epochs,'batch_size':a.batch_size,'learning_rate':a.learning_rate,'max_length':a.max_length,'seed':a.seed,'training_seconds':time.perf_counter()-start}); save_json(hist,Path(a.output_dir)/'training_history.json')
if __name__=='__main__': main()
