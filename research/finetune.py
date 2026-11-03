"""Continue training a released checkpoint with a decaying learning rate.

The released molecular models were trained with a constant LR of 4e-4 and
EMA 0.9999. This script resumes weights, EMA and Adam state and anneals the LR
with a cosine schedule, saving the EMA model periodically. Same architecture,
same parameter count.

usage: python research/finetune.py --dataset moses --steps 100000 --out DIR \
          [--lr 4e-4] [--end-lr 1e-6] [--loss kl|cross_entropy] [--ema-beta 0.9999]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Model, now  # noqa: E402

from project.utils.tensor_utils import dict_of_tensors_to_tensor_dict  # noqa: E402

def estimate_kl(task, batch, selfcond: bool, sc_prob: float) -> torch.Tensor:
    """ELBO.estimate_kl (repeat_count=1, no reweighting) plus self-conditioning."""
    belief = task.belief
    B = batch.batch_size
    t = torch.rand(B, device=task.device)
    beliefs = belief.sample_initial_belief(B)
    cls = task.class_conditioner.classify_sample(batch)
    task.class_conditioner.write_sample_classifications_to_belief(cls, beliefs)
    belief.sample_belief_update_inplace(beliefs, batch, belief.get_precision(t))
    beliefs["t"] = t
    if selfcond:
        with torch.no_grad():
            rec0 = task.reconstructor(beliefs)
        use = (torch.rand(B, device=task.device) < sc_prob).float()
        beliefs["X_sc"] = rec0["X"] * use.view(-1, 1, 1)
        beliefs["E_sc"] = rec0["E"] * use.view(-1, 1, 1, 1)
    rec = task.reconstructor(beliefs)
    return belief.dKL_dt(beliefs, batch, rec, t).mean()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--end-lr", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--loss", default=None, choices=[None, "kl", "cross_entropy"])
    p.add_argument("--ema-beta", type=float, default=None)
    p.add_argument("--save-every", type=int, default=20000)
    p.add_argument("--fresh-optimizer", action="store_true")
    p.add_argument("--selfcond", action="store_true", help="add self-conditioning inputs")
    p.add_argument("--sc-prob", type=float, default=0.5)
    p.add_argument("--rrwp-sc", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(a.dataset, a.ckpt, selfcond=a.selfcond, rrwp_sc=a.rrwp_sc)
    if model.selfcond:
        a.fresh_optimizer = True  # input layers changed shape
    task = model.task
    task.train()
    dev = model.device
    if a.loss:
        for b in model.belief.beliefs:
            b.loss_mode = a.loss
    if a.ema_beta:
        task.ema_reconstructor.beta = a.ema_beta

    opt = torch.optim.Adam(task.reconstructor.parameters(), lr=a.lr)
    state = torch.load(model.ckpt, map_location="cpu", weights_only=False)
    if not a.fresh_optimizer:
        opt.load_state_dict(state["optimizer_states"][0])
        for g in opt.param_groups:
            g["lr"] = a.lr
    del state

    def lr_at(step: int) -> float:
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        x = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.end_lr + 0.5 * (a.lr - a.end_lr) * (1 + math.cos(math.pi * min(1.0, x)))

    dm = model.datamodule
    if a.batch_size:
        dm.batch_size = a.batch_size
    step = 0
    t0 = time.time()
    run_loss = 0.0
    while step < a.steps:
        for raw in dm.train_dataloader():
            if step >= a.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = dict_of_tensors_to_tensor_dict({k: v.to(dev, non_blocking=True) for k, v in raw.items()})
            kl = estimate_kl(task, batch, model.selfcond, a.sc_prob)
            loss = kl / model.belief.get_dimentionality()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task.reconstructor.parameters(), 0.5)
            opt.step()
            task.ema_reconstructor.update()
            run_loss += loss.item()
            step += 1
            if step % 1000 == 0:
                print(f"[{now()}] step {step} loss {run_loss / 1000:.5f} lr {lr_at(step):.2e} "
                      f"{(time.time() - t0) / step:.3f}s/step", flush=True)
                run_loss = 0.0
            if step % a.save_every == 0 or step == a.steps:
                ck = {"state_dict": task.state_dict(), "optimizer_states": [opt.state_dict()],
                      "global_step": step, "args": vars(a), "selfcond": model.selfcond,
                      "rrwp_sc": model.rrwp_sc}
                torch.save(ck, out / f"step_{step}.ckpt")
                print(f"[{now()}] saved step_{step}.ckpt", flush=True)
