import json,random
from pathlib import Path
import numpy as np, torch

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def select_device(requested="auto"): return torch.device("cuda" if requested=="auto" and torch.cuda.is_available() else "cpu" if requested=="auto" else requested)
def save_json(data,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2))
def directory_size_bytes(path): return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
