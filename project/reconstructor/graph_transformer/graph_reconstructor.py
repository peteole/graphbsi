import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn.functional import softmax

from project.data.molecule import MoleculeDataModule
from project.data.synthetic_graphs import SyntheticGraphDataModule
from project.reconstructor.graph_transformer.edge_features import (
    EdgeFeaturesExtractor,
)
from project.reconstructor.graph_transformer.global_features import (
    GlobalFeaturesExtractor,
)
from project.reconstructor.graph_transformer.graph_utils import PlaceHolder, get_E_mask
from project.reconstructor.graph_transformer.node_features import (
    NodeFeaturesExtractor,
)
from project.reconstructor.graph_transformer.transformer import (
    GraphDims,
    GraphTransformer,
    HiddenDims,
)
from project.reconstructor.reconstructor import SampleReconstructor


class GraphReconstructor(SampleReconstructor):
    @classmethod
    def from_config_synthetic_graphs(
        cls,
        data_module: SyntheticGraphDataModule,
        node_features: list[NodeFeaturesExtractor],
        edge_features: list[EdgeFeaturesExtractor],
        global_features: list[GlobalFeaturesExtractor],
        num_layers: int = 6,
        hidden_mlp_dims=GraphDims(
            X=256,
            E=128,
            y=128,
        ),
        hidden_dims=HiddenDims(
            dx=128,
            de=64,
            dy=128,
            n_head=8,
            dim_ffX=256,
            dim_ffE=64,
            dim_ffy=256,
        ),
        act_fn_in: nn.Module = nn.ReLU(),
        act_fn_out: nn.Module = nn.ReLU(),
        **kwargs,
    ):
        num_node_features = 2
        num_edge_features = 2
        return cls(
            node_features_extractors=node_features,
            edge_features_extractors=edge_features,
            global_features_extractors=global_features,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_graph_features=1,  # Assuming one global feature
            num_layers=num_layers,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
        )

    @classmethod
    def from_config(
        cls,
        data_module: MoleculeDataModule,
        node_features: list[NodeFeaturesExtractor],
        edge_features: list[EdgeFeaturesExtractor],
        global_features: list[GlobalFeaturesExtractor],
        num_layers: int = 6,
        hidden_mlp_dims=GraphDims(
            X=256,
            E=128,
            y=128,
        ),
        hidden_dims=HiddenDims(
            dx=128,
            de=64,
            dy=128,
            n_head=8,
            dim_ffX=256,
            dim_ffE=64,
            dim_ffy=256,
        ),
        act_fn_in: nn.Module = nn.ReLU(),
        act_fn_out: nn.Module = nn.ReLU(),
        take_argmax=False,
        **kwargs,
    ):
        data_module.prepare_data()
        num_node_features = data_module.get_num_node_features()
        num_edge_features = data_module.get_num_edge_features()
        return cls(
            node_features_extractors=node_features,
            edge_features_extractors=edge_features,
            global_features_extractors=global_features,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_graph_features=1,
            num_layers=num_layers,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            take_argmax=take_argmax,
        )

    def __init__(
        self,
        node_features_extractors: list[NodeFeaturesExtractor],
        edge_features_extractors: list[EdgeFeaturesExtractor],
        global_features_extractors: list[GlobalFeaturesExtractor],
        num_layers: int = 6,
        num_node_features: int = 4,
        num_edge_features: int = 4,
        num_graph_features: int = 1,
        act_fn_in: nn.Module = nn.ReLU(),
        act_fn_out: nn.Module = nn.ReLU(),
        hidden_mlp_dims=GraphDims(
            X=256,
            E=128,
            y=128,
        ),
        hidden_dims=HiddenDims(
            dx=128,
            de=64,
            dy=128,
            n_head=8,
            dim_ffX=256,
            dim_ffE=64,
            dim_ffy=256,
        ),
        take_argmax: bool = False,
    ) -> None:
        super().__init__()

        self.transformer = GraphTransformer(
            n_layers=num_layers,
            input_dims=GraphDims(
                X=sum(fe.out_dim for fe in node_features_extractors),
                E=sum(fe.out_dim for fe in edge_features_extractors),
                y=sum(fe.out_dim for fe in global_features_extractors),
            ),
            output_dims=GraphDims(
                X=num_node_features,
                E=num_edge_features,
                y=num_graph_features,
            ),
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
        )
        self.node_features_extractors = nn.ModuleList(node_features_extractors)
        self.edge_features_extractors = nn.ModuleList(edge_features_extractors)
        self.global_features_extractors = nn.ModuleList(global_features_extractors)
        self.take_argmax = take_argmax

    def forward(self, belief: TensorDict) -> TensorDict:
        if belief.device.type not in ["mps", "cuda"]:
            print(
                "Warning: GraphReconstructor is not optimized for CPU. "
                "Performance may be significantly slower."
            )
        # belief is a TensorDict with keys "X", "E", "t", "
        mask = belief["mask"]
        X = torch.cat([fe(belief) for fe in self.node_features_extractors], dim=-1)
        E = torch.cat([fe(belief) for fe in self.edge_features_extractors], dim=-1)
        y = torch.cat([fe(belief) for fe in self.global_features_extractors], dim=-1)
        # X = softmax(belief["X"], dim=-1)
        # E = softmax(belief["E"], dim=-1)
        E = (E + E.transpose(1, 2)) / 2
        if self.take_argmax:
            X = torch.zeros_like(X).scatter_(
                -1, torch.argmax(X, dim=-1, keepdim=True), 1.0
            )
            E = torch.zeros_like(E).scatter_(
                -1, torch.argmax(E, dim=-1, keepdim=True), 1.0
            )
        # y = belief["t"].unsqueeze(-1)
        ph_input = PlaceHolder(X, E, y).mask(mask)
        p: PlaceHolder = self.transformer(ph_input.X, ph_input.E, ph_input.y, mask)
        X, E, y = p.X, p.E, p.y
        # E_mask2 = E.sum(dim=-1) != 0
        E_mask = get_E_mask(mask)
        # assert torch.all(E_mask == E_mask2), "E_mask and E_mask2 are not equal"

        # compute softmax over the last dimension
        atom_type_probs_raw = X[..., 1:].clone()
        atom_type_probs_raw = softmax(atom_type_probs_raw, dim=-1)
        X_proc = torch.cat([X[..., :1].clone(), atom_type_probs_raw], dim=-1)
        E = softmax(E, dim=-1)
        # set first dimension to 0 at mask=True
        X_proc[..., 0] = torch.where(
            mask, torch.zeros_like(X_proc[..., 0]), torch.ones_like(X_proc[..., 0])
        )
        X_proc[..., 1:] = torch.where(
            mask.unsqueeze(-1).expand_as(X_proc[..., 1:]),
            X_proc[..., 1:],
            torch.zeros_like(X_proc[..., 1:]),
        )

        E_proc = torch.zeros_like(E)
        E_proc[..., 0] = torch.where(E_mask, E[..., 0], torch.ones_like(E[..., 0]))
        E_proc[..., 1:] = torch.where(
            E_mask.unsqueeze(-1).expand_as(E[..., 1:]),
            E[..., 1:],
            torch.zeros_like(E[..., 1:]),
        )
        assert torch.allclose(
            X_proc.sum(dim=-1), torch.ones_like(X_proc.sum(dim=-1)), rtol=1e-5, atol=1e-5
        ), "X is not a valid one-hot encoding"
        assert torch.allclose(
            E_proc.sum(dim=-1), torch.ones_like(E_proc.sum(dim=-1)), rtol=1e-5, atol=1e-5
        ), "E is not a valid one-hot encoding"
        return TensorDict(
            {"X": X_proc, "E": E_proc, "mask": mask},
            batch_size=belief.batch_size,
            device=belief.device,
        )
