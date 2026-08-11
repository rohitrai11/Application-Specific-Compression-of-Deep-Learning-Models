#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig,AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from common.data import load_snli
from common.engine import evaluate,save_predictions
from common.modeling import NLIClassifier
from common.utils import save_json
def dynamic_quantize(model):
    try:q=torch.ao.quantization.quantize_dynamic
    except AttributeError:q=torch.quantization.quantize_dynamic
    return q(model.cpu().eval(),{nn.Linear},dtype=torch.qint8)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input_model_path',default='checkpoints/baseline2_quantized_bert'); p.add_argument('--dataset_dir',default='snli_1.0'); p.add_argument('--output_dir',default='predictions/baseline2_quantized_bert'); p.add_argument('--batch_size',type=int,default=32); p.add_argument('--max_length',type=int,default=512); a=p.parse_args(); ck=Path(a.input_model_path); model=dynamic_quantize(NLIClassifier(AutoModel.from_config(AutoConfig.from_pretrained(ck)),'google-bert/bert-base-uncased')); model.load_state_dict(torch.load(ck/'quantized_state_dict.pt',map_location='cpu')); tok=AutoTokenizer.from_pretrained(ck); data=load_snli(Path(a.dataset_dir)/'snli_1.0_test.txt',tok,a.max_length); r=evaluate(model,DataLoader(data,batch_size=a.batch_size),torch.device('cpu'),'test'); save_predictions(r,a.output_dir,'snli_quantized_bert'); m={k:v for k,v in r.items() if k not in {'predictions','probabilities'}}; mp=ck/'metadata.json'; m['checkpoint_metadata']=json.loads(mp.read_text()) if mp.exists() else {}; save_json(m,Path(a.output_dir)/'metrics.json'); print(f"SNLI test accuracy: {r['accuracy']:.6f}")
if __name__=='__main__': main()
