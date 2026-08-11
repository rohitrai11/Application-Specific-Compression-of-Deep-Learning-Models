import json,os,random
from pathlib import Path
import numpy as np,torch
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def select_device(choice='auto'):
    if choice=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is not available.')
    return torch.device('cuda' if choice=='auto' and torch.cuda.is_available() else 'cpu' if choice=='auto' else choice)
def save_json(payload,path): path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2))
def directory_size_bytes(path): return sum(os.path.getsize(os.path.join(r,n)) for r,_,fs in os.walk(path) for n in fs)
