from __future__ import annotations
import pickle
from pathlib import Path
import data_loader  # noqa: F401
def read_dataset(path):
 with Path(path).open('rb') as inp: return pickle.load(inp)
