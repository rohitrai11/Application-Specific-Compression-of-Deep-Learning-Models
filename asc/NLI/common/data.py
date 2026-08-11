"""SNLI data loading used by the NLI baseline experiments.

This module follows the label mapping and SNLI text-column selection used in the
reference NLI code:
    contradiction -> 0
    neutral       -> 1
    entailment    -> 2

Expected files:
    snli_1.0/snli_1.0_train.txt
    snli_1.0/snli_1.0_dev.txt
    snli_1.0/snli_1.0_test.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

LABEL_TO_ID = {"contradiction": 0, "neutral": 1, "entailment": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def _wordpiece_tokenize(sentence: str, tokenizer) -> List[str]:
    pieces: List[str] = []
    for word in str(sentence).strip().split():
        pieces.extend(tokenizer.tokenize(word))
    return pieces


def _truncate_pair(tokens_a: List[str], tokens_b: List[str], max_tokens: int) -> None:
    while len(tokens_a) + len(tokens_b) > max_tokens:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class SNLIDataset(Dataset):
    def __init__(self, sentence1, sentence2, labels, tokenizer, max_length: int = 512):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent1 = _wordpiece_tokenize(self.sentence1[idx], self.tokenizer)
        sent2 = _wordpiece_tokenize(self.sentence2[idx], self.tokenizer)
        _truncate_pair(sent1, sent2, self.max_length - 3)
        cls = self.tokenizer.cls_token or "[CLS]"
        sep = self.tokenizer.sep_token or "[SEP]"
        pad = self.tokenizer.pad_token or "[PAD]"
        tokens = [cls] + sent1 + [sep] + sent2 + [sep]
        token_type_ids = [0] * (len(sent1) + 2) + [1] * (len(sent2) + 1)
        attention_mask = [1] * len(tokens)
        pad_len = self.max_length - len(tokens)
        tokens += [pad] * pad_len
        token_type_ids += [0] * pad_len
        attention_mask += [0] * pad_len
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        label = LABEL_TO_ID[self.labels[idx]]
        return {"index": torch.tensor(idx), "ids": torch.tensor(ids), "mask": torch.tensor(attention_mask), "token_type_ids": torch.tensor(token_type_ids), "target": torch.tensor(label)}


def load_snli(file_path: str | Path, tokenizer, max_length: int = 512) -> SNLIDataset:
    sentence1_list=[]; sentence2_list=[]; target_label_list=[]
    file_path=Path(file_path)
    with file_path.open("r",encoding="utf-8") as file:
        for line in file:
            parts=line.rstrip("\n").split("\t")
            if len(parts)<9 or parts[0] not in LABEL_TO_ID: continue
            target_label_list.append(parts[0]); sentence1_list.append(parts[-9]); sentence2_list.append(parts[-8])
    print(f"Loaded {len(target_label_list):,} examples from {file_path}")
    return SNLIDataset(sentence1_list,sentence2_list,target_label_list,tokenizer,max_length)
