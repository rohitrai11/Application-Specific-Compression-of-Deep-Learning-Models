from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig,AutoModel,AutoTokenizer

class NLIClassifier(nn.Module):
    def __init__(self,backbone,model_id=""):
        super().__init__(); self.backbone=backbone; self.model_id=model_id; h=int(backbone.config.hidden_size); self.hidden=nn.Linear(h,6); self.classifier=nn.Linear(6,3)
    @classmethod
    def from_pretrained_model(cls,model_id): return cls(AutoModel.from_pretrained(model_id),model_id)
    def forward(self,input_ids,attention_mask,token_type_ids=None,labels=None):
        kw={"input_ids":input_ids,"attention_mask":attention_mask}
        if token_type_ids is not None and self.backbone.config.model_type!="distilbert": kw["token_type_ids"]=token_type_ids
        x=self.backbone(**kw).last_hidden_state[:,0,:]; logits=self.classifier(self.hidden(x)); p=F.softmax(logits,dim=1)
        return (F.cross_entropy(logits,labels.view(-1)) if labels is not None else None),p

def save_float_checkpoint(model,tokenizer,output_dir,metadata=None):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); model.backbone.save_pretrained(out/"backbone"); tokenizer.save_pretrained(out)
    torch.save({"hidden":model.hidden.state_dict(),"classifier":model.classifier.state_dict()},out/"classifier_head.pt")
    payload={"model_id":model.model_id}; payload.update(metadata or {}); (out/"metadata.json").write_text(json.dumps(payload,indent=2))

def load_float_checkpoint(checkpoint_dir,map_location="cpu"):
    d=Path(checkpoint_dir); m=json.loads((d/"metadata.json").read_text()) if (d/"metadata.json").exists() else {}; model=NLIClassifier(AutoModel.from_pretrained(d/"backbone"),m.get("model_id",""))
    try: head=torch.load(d/"classifier_head.pt",map_location=map_location,weights_only=True)
    except TypeError: head=torch.load(d/"classifier_head.pt",map_location=map_location)
    model.hidden.load_state_dict(head["hidden"]); model.classifier.load_state_dict(head["classifier"]); return model,AutoTokenizer.from_pretrained(d),m

def build_model_from_local_config(config_dir): return NLIClassifier(AutoModel.from_config(AutoConfig.from_pretrained(config_dir)))
