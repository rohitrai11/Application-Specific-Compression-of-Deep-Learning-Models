"""Shared Application Specific Compression (ASC) utilities.

The layer-selection routine follows the supplied notebook: hidden-state index 0
is the embedding output; indices 1..12 are BERT encoder outputs. Later states
whose cosine similarity with a surviving earlier state reaches the effective
threshold are marked redundant. The notebook uses threshold-0.01, exposed here
as epsilon. Layer selection uses the dataset-averaged matrix and masks padding.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def compute_dataset_similarity(backbone,dataloader,device,prepare_inputs,max_samples=None):
    backbone.eval().to(device); total=None; count=0
    for batch in tqdm(dataloader,desc='ASC similarity',ncols=100):
        inputs,mask=prepare_inputs(batch,device); hs=backbone(**inputs,output_hidden_states=True,return_dict=True).hidden_states
        states=F.normalize(torch.stack(tuple(hs),0).float(),p=2,dim=-1); pair=torch.einsum('sblh,tblh->stbl',states,states); mask=mask.to(device).bool()
        for b in range(states.shape[1]):
            valid=mask[b]
            if not valid.any(): continue
            sample=pair[:,:,b,valid].mean(-1); total=torch.zeros_like(sample) if total is None else total; total+=sample; count+=1
            if max_samples is not None and count>=max_samples: break
        if max_samples is not None and count>=max_samples: break
    if not count: raise RuntimeError('No examples were processed while computing similarity.')
    m=total/float(count); m=.5*(m+m.T); m.fill_diagonal_(1.); return m.cpu(),count

def select_layers_notebook_logic(similarity_matrix,threshold,epsilon=.01):
    m=torch.as_tensor(similarity_matrix,dtype=torch.float32); effective=float(threshold)-float(epsilon); redundant=set()
    for i in range(m.shape[0]):
        if i in redundant: continue
        for k in range(i+1,m.shape[0]):
            if k not in redundant and float(m[i,k])>=effective: redundant.add(k)
    enc=sorted(h-1 for h in redundant if h>=1)
    return {'requested_threshold':float(threshold),'epsilon':float(epsilon),'effective_threshold':effective,'redundant_hidden_state_indices':sorted(redundant),'encoder_indices_zero_based':enc,'encoder_layers_one_based':[i+1 for i in enc]}

def prune_bert_encoder(backbone,encoder_indices_zero_based):
    layers=list(backbone.encoder.layer); remove=sorted(set(map(int,encoder_indices_zero_based))); retained=[i for i in range(len(layers)) if i not in set(remove)]
    if not retained: raise ValueError('ASC selection would remove every encoder layer.')
    backbone.encoder.layer=nn.ModuleList([layers[i] for i in retained]); backbone.config.num_hidden_layers=len(retained)
    return {'original_num_hidden_layers':len(layers),'removed_encoder_indices_zero_based':remove,'removed_encoder_layers_one_based':[i+1 for i in remove],'retained_encoder_indices_zero_based':retained,'retained_encoder_layers_one_based':[i+1 for i in retained],'num_hidden_layers_after_compression':len(retained)}

def save_similarity_bundle(similarity_matrix,output_dir,prefix,sample_count,labels=None):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); m=torch.as_tensor(similarity_matrix).float().cpu(); labels=labels or ['embed']+[f'enc{i}' for i in range(1,m.shape[0])]; pt=out/f'{prefix}.pt'; npy=out/f'{prefix}.npy'; csvp=out/f'{prefix}.csv'; js=out/f'{prefix}.json'; torch.save(m,pt); np.save(npy,m.numpy())
    with csvp.open('w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['']+list(labels)); [w.writerow([lab]+[f'{x:.10f}' for x in row]) for lab,row in zip(labels,m.tolist())]
    js.write_text(json.dumps({'sample_count':int(sample_count),'labels':list(labels),'matrix':m.tolist()},indent=2)); return {'pt':str(pt),'npy':str(npy),'csv':str(csvp),'json':str(js)}
def load_similarity(path):
    p=Path(path)
    if p.suffix=='.pt': return torch.load(p,map_location='cpu')
    if p.suffix=='.npy': return torch.from_numpy(np.load(p)).float()
    if p.suffix=='.json': return torch.tensor(json.loads(p.read_text())['matrix']).float()
    raise ValueError('Similarity path must end in .pt, .npy, or .json')
def save_json(data,path): p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2))
def threshold_tag(threshold): return str(int(round(float(threshold)*100)))
