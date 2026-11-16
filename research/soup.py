"""Average the weights of several checkpoints of the same model ("model soup").

Parameter count is unchanged, so this is admissible under a fixed-parameter
budget. GuacaMol has three released checkpoints 30 epochs apart, which is far
outside the horizon of the EMA used during training (beta=0.9999 averages over
roughly 10 000 steps, well under one epoch), so the average is genuinely
different from any of them.

usage: python research/soup.py --out DIR/soup.ckpt CKPT [CKPT ...]
"""

import argparse
from pathlib import Path

import torch

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", help="comma-separated mixing weights, default uniform")
    a = p.parse_args()

    ws = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(a.ckpts)
    assert len(ws) == len(a.ckpts)
    total = sum(ws)
    ws = [w / total for w in ws]

    out: dict[str, torch.Tensor] | None = None
    meta = None
    for path, w in zip(a.ckpts, ws):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        if out is None:
            out = {k: torch.zeros_like(v, dtype=torch.float64) if v.is_floating_point() else v.clone()
                   for k, v in sd.items()}
            meta = ck
        for k, v in sd.items():
            if v.is_floating_point():
                out[k] += w * v.double()
    assert out is not None and meta is not None
    sd0 = meta["state_dict"]
    meta["state_dict"] = {k: (v.to(sd0[k].dtype) if sd0[k].is_floating_point() else v)
                          for k, v in out.items()}
    meta["soup_of"] = a.ckpts
    meta["soup_weights"] = ws
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta, a.out)
    print(f"wrote {a.out} from {len(a.ckpts)} checkpoints, weights {ws}")
