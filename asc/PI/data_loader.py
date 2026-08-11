import torch,pandas as pd,numpy as np
from torch.utils.data import Dataset
MAX_LEN=512
def tokenize_sent(sentence,tokenizer):
    out=[]
    for word in str(sentence).strip().split(): out.extend(tokenizer.tokenize(word))
    return out
class qqp_dataset(Dataset):
    def __init__(self,sentence1,sentence2,label,tokenizer,max_len): self.len=len(sentence1); self.sentence1=sentence1; self.sentence2=sentence2; self.label=label; self.max_len=max_len; self.tokenizer=tokenizer
    def __len__(self): return self.len
    def __getitem__(self,idx):
        s1=tokenize_sent(self.sentence1[idx],self.tokenizer); s2=tokenize_sent(self.sentence2[idx],self.tokenizer); tokens=['[CLS]']+s1+['[SEP]']+s2+['[SEP]']; tt=[0]*(len(s1)+2)+[1]*(len(s2)+1); pad=self.max_len-len(tokens); tokens+=['[PAD]']*pad; tt+=[0]*pad; mask=[1 if x!='[PAD]' else 0 for x in tokens]; ids=self.tokenizer.convert_tokens_to_ids(tokens); return {'index':idx,'ids':torch.tensor(ids),'mask':torch.tensor(mask),'token_type_ids':torch.tensor(tt),'target':torch.tensor([self.label[idx]])}
def load_qqp(file_path,tokenizer):
    df=pd.read_csv(file_path); return qqp_dataset(df.question1.tolist(),df.question2.tolist(),df.is_duplicate.tolist(),tokenizer,MAX_LEN)
class paws_dataset(qqp_dataset): pass
def load_paws(file_path,tokenizer):
    df=pd.read_csv(file_path,delimiter='\t'); return paws_dataset(df.sentence1.tolist(),df.sentence2.tolist(),df.label.tolist(),tokenizer,MAX_LEN)
