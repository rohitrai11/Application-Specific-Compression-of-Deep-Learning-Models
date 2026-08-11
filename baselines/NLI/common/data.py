from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
LABEL_TO_ID={'contradiction':0,'neutral':1,'entailment':2}; ID_TO_LABEL={v:k for k,v in LABEL_TO_ID.items()}
def _tok(s,t):
 out=[]
 for w in str(s).strip().split(): out.extend(t.tokenize(w))
 return out
def _trunc(a,b,n):
 while len(a)+len(b)>n: (a if len(a)>len(b) else b).pop()
class SNLIDataset(Dataset):
 def __init__(self,s1,s2,y,t,max_length=512): self.s1=s1; self.s2=s2; self.y=y; self.t=t; self.max_length=max_length
 def __len__(self): return len(self.y)
 def __getitem__(self,i):
  a=_tok(self.s1[i],self.t); b=_tok(self.s2[i],self.t); _trunc(a,b,self.max_length-3); tok=[self.t.cls_token]+a+[self.t.sep_token]+b+[self.t.sep_token]; tt=[0]*(len(a)+2)+[1]*(len(b)+1); mask=[1]*len(tok); p=self.max_length-len(tok); tok += [self.t.pad_token]*p; tt += [0]*p; mask += [0]*p; return {'index':torch.tensor(i),'ids':torch.tensor(self.t.convert_tokens_to_ids(tok)),'mask':torch.tensor(mask),'token_type_ids':torch.tensor(tt),'target':torch.tensor(LABEL_TO_ID[self.y[i]])}
def load_snli(file_path,tokenizer,max_length=512):
 s1=[];s2=[];y=[]
 with Path(file_path).open(encoding='utf-8') as f:
  for line in f:
   x=line.rstrip('\n').split('\t')
   if len(x)>=9 and x[0] in LABEL_TO_ID: y.append(x[0]);s1.append(x[-9]);s2.append(x[-8])
 return SNLIDataset(s1,s2,y,tokenizer,max_length)
