from __future__ import annotations
import json
from pathlib import Path
import torch,torch.nn as nn,torch.nn.functional as F
from transformers import AutoConfig,AutoModel,AutoTokenizer
class PIClassifier(nn.Module):
 def __init__(self,backbone,model_id=''):
  super().__init__(); self.backbone=backbone; self.model_id=model_id; self.classifier=nn.Linear(int(backbone.config.hidden_size),2)
 @classmethod
 def from_pretrained_model(cls,model_id): return cls(AutoModel.from_pretrained(model_id),model_id)
 def forward(self,input_ids,attention_mask,token_type_ids=None,labels=None):
  kw={'input_ids':input_ids,'attention_mask':attention_mask}
  if token_type_ids is not None and self.backbone.config.model_type!='distilbert': kw['token_type_ids']=token_type_ids
  logits=self.classifier(self.backbone(**kw).last_hidden_state[:,0,:]); probs=F.softmax(logits,dim=1); loss=None
  if labels is not None: loss=-torch.log(torch.gather(probs,1,labels.view(-1,1))).mean()
  return loss,probs
def save_float_checkpoint(model,tokenizer,output_dir,metadata=None):
 out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); model.backbone.save_pretrained(out/'backbone'); tokenizer.save_pretrained(out); torch.save(model.classifier.state_dict(),out/'classifier_head.pt'); payload={'model_id':model.model_id}; payload.update(metadata or {}); (out/'metadata.json').write_text(json.dumps(payload,indent=2))
def load_float_checkpoint(checkpoint_dir,map_location='cpu'):
 d=Path(checkpoint_dir); meta=json.loads((d/'metadata.json').read_text()) if (d/'metadata.json').exists() else {}; model=PIClassifier(AutoModel.from_pretrained(d/'backbone'),meta.get('model_id','')); head=torch.load(d/'classifier_head.pt',map_location=map_location,weights_only=True); model.classifier.load_state_dict(head); return model,AutoTokenizer.from_pretrained(d),meta
def build_model_from_local_config(config_dir): return PIClassifier(AutoModel.from_config(AutoConfig.from_pretrained(config_dir)))
