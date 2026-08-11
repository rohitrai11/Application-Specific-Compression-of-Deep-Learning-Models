#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,torch
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]; TASK=Path(__file__).resolve().parent; sys.path[:0]=[str(REPO),str(TASK)]
from common.asc_utils import compute_dataset_similarity,save_similarity_bundle
from NLI.common.data import load_snli
from NLI.common.modeling import load_float_checkpoint
from NLI.common.utils import select_device

def prepare_inputs(batch,device):
    x={"input_ids":batch["ids"].to(device),"attention_mask":batch["mask"].to(device),"token_type_ids":batch["token_type_ids"].to(device)}; return x,x["attention_mask"]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input_model_path",default="NLI/checkpoints/bert_base_finetuned"); p.add_argument("--dataset_dir",default="NLI/snli_1.0"); p.add_argument("--output_dir",default="NLI/similarity"); p.add_argument("--batch_size",type=int,default=1); p.add_argument("--max_length",type=int,default=512); p.add_argument("--max_samples",type=int); p.add_argument("--device",default="auto"); a=p.parse_args()
    device=select_device(a.device); model,tok,_=load_float_checkpoint(a.input_model_path); model.to(device).eval(); data=load_snli(Path(a.dataset_dir)/"snli_1.0_train.txt",tok,a.max_length); matrix,count=compute_dataset_similarity(model.backbone,DataLoader(data,batch_size=a.batch_size),device,prepare_inputs,a.max_samples); print(save_similarity_bundle(matrix,a.output_dir,"nli_similarity_matrix",count))
if __name__=="__main__": main()
