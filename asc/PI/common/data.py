from __future__ import annotations
import pickle,sys
from pathlib import Path
PI_ROOT=Path(__file__).resolve().parents[1]
if str(PI_ROOT) not in sys.path: sys.path.insert(0,str(PI_ROOT))
import data_loader  # noqa: F401
def read_dataset(path):
    with Path(path).open('rb') as inp: return pickle.load(inp)
