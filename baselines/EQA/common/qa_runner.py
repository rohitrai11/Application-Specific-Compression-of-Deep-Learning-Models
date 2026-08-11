#!/usr/bin/env python3
"""Shared SQuAD2.0 training/evaluation utilities for EQA baselines and ASC."""
from __future__ import annotations
import collections, json, math, random, re, string, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_squad_examples(path, answerable_only=False):
    data=json.loads(Path(path).read_text(encoding='utf-8')); out=[]
    for article in data['data']:
        for para in article['paragraphs']:
            for qa in para['qas']:
                ans=qa.get('answers',[]); impossible=bool(qa.get('is_impossible',False)) or not ans
                if answerable_only and impossible: continue
                out.append({'id':str(qa['id']),'question':qa['question'],'context':para['context'],'answers':ans,'is_impossible':impossible})
    return out

def _short_question(q, tokenizer, n):
    ids=tokenizer(q,add_special_tokens=False,truncation=True,max_length=n)['input_ids']; return tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)

def build_features(examples, tokenizer, max_seq_length=384, doc_stride=128, max_query_length=64, is_training=False):
    features=[]
    for ei,ex in enumerate(tqdm(examples,desc='Tokenizing')):
        enc=tokenizer(_short_question(ex['question'],tokenizer,max_query_length),ex['context'],truncation='only_second',max_length=max_seq_length,stride=doc_stride,return_overflowing_tokens=True,return_offsets_mapping=True,padding='max_length')
        for si,input_ids in enumerate(enc['input_ids']):
            seq=enc.sequence_ids(si); offsets=enc['offset_mapping'][si]
            f={'input_ids':input_ids,'attention_mask':enc['attention_mask'][si],'example_index':ei,'example_id':ex['id'],'offset_mapping':[tuple(o) if s==1 else None for s,o in zip(seq,offsets)]}
            if 'token_type_ids' in enc: f['token_type_ids']=enc['token_type_ids'][si]
            if is_training:
                try: cls=input_ids.index(tokenizer.cls_token_id)
                except ValueError: cls=0
                start=end=cls
                if not ex['is_impossible'] and ex['answers']:
                    a=ex['answers'][0]; sc=int(a['answer_start']); ec=sc+len(a['text']); ci=[i for i,s in enumerate(seq) if s==1]
                    if ci and offsets[ci[0]][0] <= sc and offsets[ci[-1]][1] >= ec:
                        s=ci[0]
                        while s<=ci[-1] and offsets[s][0] <= sc: s+=1
                        e=ci[-1]
                        while e>=ci[0] and offsets[e][1] >= ec: e-=1
                        start,end=s-1,e+1
                f['start_positions']=start; f['end_positions']=end
            features.append(f)
    return features

class QATensorDataset(Dataset):
    def __init__(self, features, is_training): self.features=features; self.is_training=is_training
    def __len__(self): return len(self.features)
    def __getitem__(self,i):
        f=self.features[i]; x={'input_ids':torch.tensor(f['input_ids'],dtype=torch.long),'attention_mask':torch.tensor(f['attention_mask'],dtype=torch.long),'feature_index':torch.tensor(i,dtype=torch.long)}
        if 'token_type_ids' in f: x['token_type_ids']=torch.tensor(f['token_type_ids'],dtype=torch.long)
        if self.is_training: x['start_positions']=torch.tensor(f['start_positions'],dtype=torch.long); x['end_positions']=torch.tensor(f['end_positions'],dtype=torch.long)
        return x

def _inputs(batch,device):
    keys={'input_ids','attention_mask','token_type_ids','start_positions','end_positions'}; return {k:v.to(device) for k,v in batch.items() if k in keys}

def train_model(model, tokenizer, train_examples, output_dir, device, epochs=3, batch_size=32, learning_rate=3e-5, warmup_ratio=.1, weight_decay=0., max_seq_length=384, doc_stride=128, max_query_length=64, seed=42):
    fs=build_features(train_examples,tokenizer,max_seq_length,doc_stride,max_query_length,True); dl=DataLoader(QATensorDataset(fs,True),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); no_decay=('bias','LayerNorm.weight'); groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and not any(x in n for x in no_decay)],'weight_decay':weight_decay},{'params':[p for n,p in model.named_parameters() if p.requires_grad and any(x in n for x in no_decay)],'weight_decay':0.}]; opt=torch.optim.AdamW(groups,lr=learning_rate,eps=1e-8); steps=max(1,int(math.ceil(len(dl)*epochs))); warm=int(steps*warmup_ratio); sched=get_linear_schedule_with_warmup(opt,warm,steps); model.to(device); done=0
    for ep in range(int(math.ceil(epochs))):
        for b in tqdm(dl,desc=f'Epoch {ep+1}'):
            model.train(); opt.zero_grad(set_to_none=True); loss=model(**_inputs(b,device)).loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); done+=1
            if done>=steps: break
        if done>=steps: break
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); tokenizer.save_pretrained(out); return {'train_examples':len(train_examples),'train_features':len(fs),'global_steps':done,'warmup_steps':warm}

def normalize_answer(s):
    s=s.lower(); s=''.join(c for c in s if c not in set(string.punctuation)); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def _exact(g,p): return int(normalize_answer(g)==normalize_answer(p))
def _f1(g,p):
    gt=normalize_answer(g).split(); pt=normalize_answer(p).split(); same=sum((collections.Counter(gt)&collections.Counter(pt)).values())
    if not gt or not pt: return float(gt==pt)
    if not same: return 0.
    pr=same/len(pt); rc=same/len(gt); return 2*pr*rc/(pr+rc)
def postprocess_predictions(examples,features,start_logits,end_logits,n_best_size=20,max_answer_length=30):
    by=collections.defaultdict(list)
    for i,f in enumerate(features): by[f['example_index']].append(i)
    pred={}
    for ei,ex in enumerate(examples):
        best=(-float('inf'),'')
        for fi in by[ei]:
            off=features[fi]['offset_mapping']; sl=start_logits[fi]; el=end_logits[fi]
            for s in np.argsort(sl)[-n_best_size:][::-1]:
                for e in np.argsort(el)[-n_best_size:][::-1]:
                    if s>=len(off) or e>=len(off) or off[s] is None or off[e] is None or e<s or e-s+1>max_answer_length: continue
                    score=float(sl[s]+el[e])
                    if score>best[0]: best=(score,ex['context'][off[s][0]:off[e][1]])
        pred[ex['id']]=best[1]
    return pred
def evaluate_predictions(examples,predictions):
    em=[]; f1=[]
    for ex in examples:
        p=predictions.get(ex['id'],''); gold=[a['text'] for a in ex['answers'] if normalize_answer(a.get('text',''))] or ['']; em.append(max(_exact(g,p) for g in gold)); f1.append(max(_f1(g,p) for g in gold))
    n=max(1,len(examples)); return {'exact':100*sum(em)/n,'f1':100*sum(f1)/n,'total':len(examples)}
def evaluate_model(model,tokenizer,eval_examples,output_dir,device,batch_size=32,max_seq_length=384,doc_stride=128,max_query_length=64,n_best_size=20,max_answer_length=30):
    fs=build_features(eval_examples,tokenizer,max_seq_length,doc_stride,max_query_length,False); dl=DataLoader(QATensorDataset(fs,False),batch_size=batch_size,shuffle=False); sl=[None]*len(fs); el=[None]*len(fs); model.to(device).eval(); t=time.perf_counter()
    with torch.no_grad():
        for b in tqdm(dl,desc='Evaluating'):
            idx=b['feature_index'].tolist(); o=model(**_inputs(b,device)); s=o.start_logits.cpu().numpy(); e=o.end_logits.cpu().numpy()
            for r,i in enumerate(idx): sl[i]=s[r]; el[i]=e[r]
    elapsed=time.perf_counter()-t; pred=postprocess_predictions(eval_examples,fs,sl,el,n_best_size,max_answer_length); m=evaluate_predictions(eval_examples,pred); m.update({'evaluation_seconds':elapsed,'seconds_per_feature':elapsed/max(1,len(fs)),'eval_features':len(fs)}); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/'predictions.json').write_text(json.dumps(pred,indent=2)); (out/'metrics.json').write_text(json.dumps(m,indent=2)); return m

def _remove_random_layers(model, number_to_remove, seed, explicit=None):
    from torch import nn as _nn
    layers=model.bert.encoder.layer; n_layers=len(layers)
    if explicit:
        removed=sorted({int(x.strip()) for x in explicit.split(',') if x.strip()})
        if len(removed)!=number_to_remove: raise ValueError(f'Expected {number_to_remove} layer indices, got {removed}')
    else: removed=sorted(random.Random(seed).sample(range(n_layers),number_to_remove))
    retained=[i for i in range(n_layers) if i not in removed]; model.bert.encoder.layer=_nn.ModuleList([layers[i] for i in retained]); model.config.num_hidden_layers=len(retained); return {'original_num_hidden_layers':n_layers,'removed_layers_zero_based':removed,'retained_layers_zero_based':retained,'num_hidden_layers_after_removal':len(retained),'selection_seed':seed}

def _cli_main():
    import argparse
    from transformers import AutoModelForQuestionAnswering,AutoTokenizer
    p=argparse.ArgumentParser(); p.add_argument('--model_name_or_path',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--train_file'); p.add_argument('--predict_file'); p.add_argument('--do_train',action='store_true'); p.add_argument('--do_eval',action='store_true'); p.add_argument('--answerable_only',action='store_true'); p.add_argument('--do_lower_case',action='store_true'); p.add_argument('--num_train_epochs',type=float,default=3.0); p.add_argument('--per_gpu_train_batch_size',type=int,default=32); p.add_argument('--per_gpu_eval_batch_size',type=int,default=32); p.add_argument('--learning_rate',type=float,default=3e-5); p.add_argument('--warmup_ratio',type=float,default=.1); p.add_argument('--max_seq_length',type=int,default=384); p.add_argument('--doc_stride',type=int,default=128); p.add_argument('--max_query_length',type=int,default=64); p.add_argument('--n_best_size',type=int,default=20); p.add_argument('--max_answer_length',type=int,default=30); p.add_argument('--seed',type=int,default=42); p.add_argument('--no_cuda',action='store_true'); p.add_argument('--random_remove_layers',type=int,default=0); p.add_argument('--random_layer_seed',type=int,default=42); p.add_argument('--removed_layers'); a=p.parse_args()
    if not a.do_train and not a.do_eval: raise ValueError('Specify --do_train and/or --do_eval.')
    set_seed(a.seed); device=torch.device('cpu' if a.no_cuda or not torch.cuda.is_available() else 'cuda'); load_path=a.output_dir if a.do_eval and not a.do_train and (Path(a.output_dir)/'config.json').exists() else a.model_name_or_path; tok=AutoTokenizer.from_pretrained(load_path,use_fast=True,do_lower_case=a.do_lower_case); model=AutoModelForQuestionAnswering.from_pretrained(load_path); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'run_config.json').write_text(json.dumps(vars(a),indent=2))
    if a.do_train and a.random_remove_layers: (out/'layer_removal_metadata.json').write_text(json.dumps(_remove_random_layers(model,a.random_remove_layers,a.random_layer_seed,a.removed_layers),indent=2))
    if a.do_train:
        if not a.train_file: raise ValueError('--train_file is required with --do_train.')
        info=train_model(model,tok,load_squad_examples(a.train_file,a.answerable_only),a.output_dir,device,a.num_train_epochs,a.per_gpu_train_batch_size,a.learning_rate,a.warmup_ratio,0.,a.max_seq_length,a.doc_stride,a.max_query_length,a.seed); (out/'train_info.json').write_text(json.dumps(info,indent=2))
    if a.do_eval:
        if not a.predict_file: raise ValueError('--predict_file is required with --do_eval.')
        if a.do_train: model=AutoModelForQuestionAnswering.from_pretrained(a.output_dir); tok=AutoTokenizer.from_pretrained(a.output_dir,use_fast=True)
        print(json.dumps(evaluate_model(model,tok,load_squad_examples(a.predict_file,a.answerable_only),a.output_dir,device,a.per_gpu_eval_batch_size,a.max_seq_length,a.doc_stride,a.max_query_length,a.n_best_size,a.max_answer_length),indent=2))
if __name__=='__main__': _cli_main()
