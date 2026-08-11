"""Training, validation and evaluation loops shared by NLI experiments."""
from __future__ import annotations
import time
from pathlib import Path
import torch
from tqdm import tqdm
from .data import ID_TO_LABEL


def train_one_epoch(model,dataloader,optimizer,device,log_every=100):
    model.train(); total_loss=0.; correct=0; total=0
    for step,batch in enumerate(tqdm(dataloader,desc="train",ncols=100)):
        ids=batch["ids"].to(device); mask=batch["mask"].to(device); tt=batch["token_type_ids"].to(device); y=batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True); loss,p=model(ids,mask,tt,y); loss.backward(); optimizer.step()
        total_loss+=loss.item(); pred=p.argmax(1); correct+=(pred==y).sum().item(); total+=y.numel()
        if step%log_every==0: print(f"step={step} loss_sum={total_loss:.6f} accuracy={correct/max(total,1):.6f}")
    return {"loss_sum":total_loss,"accuracy":correct/max(total,1)}


@torch.no_grad()
def evaluate(model,dataloader,device,desc="eval"):
    model.eval(); total_loss=0.; correct=0; total=0; predictions=[]; probabilities=[]; start=time.perf_counter()
    for batch in tqdm(dataloader,desc=desc,ncols=100):
        ids=batch["ids"].to(device); mask=batch["mask"].to(device); tt=batch["token_type_ids"].to(device); y=batch["target"].to(device)
        loss,p=model(ids,mask,tt,y); total_loss+=0 if loss is None else loss.item(); pred=p.argmax(1); correct+=(pred==y).sum().item(); total+=y.numel()
        predictions.extend(ID_TO_LABEL[int(x)] for x in pred.cpu()); probabilities.extend(p.cpu().tolist())
    return {"loss_sum":total_loss,"accuracy":correct/max(total,1),"elapsed_seconds":time.perf_counter()-start,"examples":total,"predictions":predictions,"probabilities":probabilities}


def save_predictions(result,output_dir,prefix):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/f"pred_{prefix}.txt").write_text("\n".join(result["predictions"])+"\n")
    with (output_dir/f"prob_{prefix}.txt").open("w") as fh:
        for row in result["probabilities"]: fh.write(" ".join(f"{p:.10g}" for p in row)+"\n")
