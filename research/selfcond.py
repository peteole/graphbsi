"""Self-conditioning for the graph reconstructor.

The reconstructor additionally sees its previous reconstruction (`X_sc`,
`E_sc` in the belief TensorDict): the atom/bond probabilities and a random-walk
encoding (RRWP) of the predicted bond graph. During sampling the previous
step's reconstruction is free, so the number of function evaluations does not
change. A released checkpoint is converted by appending zero-initialised input
columns to the first layer of the node and edge input MLPs, so the converted
model computes exactly the released function until it is trained further.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from project.reconstructor.graph_transformer.edge_features import EdgeFeaturesExtractor
from project.reconstructor.graph_transformer.node_features import NodeFeaturesExtractor


class SelfCondNodeFeatures(NodeFeaturesExtractor):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, belief) -> Tensor:
        if "X_sc" in belief.keys():
            return belief["X_sc"]
        return torch.zeros_like(belief["X"])

    @property
    def out_dim(self) -> int:
        return self.num_classes


class SelfCondEdgeFeatures(EdgeFeaturesExtractor):
    """Previous bond probabilities plus RRWP steps 1..k of the predicted graph."""

    def __init__(self, num_classes: int, rrwp_steps: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.rrwp_steps = rrwp_steps

    def forward(self, belief) -> Tensor:
        if "E_sc" not in belief.keys():
            shape = belief["E"].shape[:-1] + (self.out_dim,)
            return torch.zeros(shape, device=belief["E"].device, dtype=belief["E"].dtype)
        E_sc = belief["E_sc"]
        A = 1 - E_sc[..., 0]
        A = (A + A.transpose(1, 2)) / 2
        P = A / A.sum(-1, keepdim=True).clamp_min(1e-10)
        walks = []
        cur = P
        for _ in range(self.rrwp_steps):
            walks.append(cur)
            cur = cur @ P
        return torch.cat([E_sc, torch.stack(walks, -1)], -1)

    @property
    def out_dim(self) -> int:
        return self.num_classes + self.rrwp_steps


def _widen_first_linear(seq: nn.Sequential, extra: int) -> None:
    old: nn.Linear = seq[0]
    new = nn.Linear(old.in_features + extra, old.out_features, device=old.weight.device)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_features] = old.weight
        new.bias.copy_(old.bias)
    seq[0] = new


def add_self_conditioning(recon, num_atom_types: int, num_bond_types: int, rrwp_steps: int = 8) -> None:
    """Convert a GraphReconstructor in place (function unchanged at init)."""
    if getattr(recon, "self_conditioning", False):
        return
    nf = SelfCondNodeFeatures(num_atom_types)
    ef = SelfCondEdgeFeatures(num_bond_types, rrwp_steps)
    recon.node_features_extractors.append(nf)
    recon.edge_features_extractors.append(ef)
    _widen_first_linear(recon.transformer.mlp_in_X, nf.out_dim)
    _widen_first_linear(recon.transformer.mlp_in_E, ef.out_dim)
    recon.self_conditioning = True


def set_self_condition(belief_params, reconstruction) -> None:
    belief_params["X_sc"] = reconstruction["X"].detach()
    belief_params["E_sc"] = reconstruction["E"].detach()


def clear_self_condition(belief_params) -> None:
    for k in ("X_sc", "E_sc"):
        if k in belief_params.keys():
            del belief_params[k]
