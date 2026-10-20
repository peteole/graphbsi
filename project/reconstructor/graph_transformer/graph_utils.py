import torch
import torch.nn as nn

from project.utils.tensor_utils import expand_tensor

# from torchdiffeq import odeint


def assert_correctly_masked(variable, node_mask):
    masking_error = (variable * (1 - node_mask.long())).abs().max().item()
    if masking_error > 1e-4:
        print(f"Masking error: {masking_error}")
        print(f"Variable: {variable}")
        print(f"Node mask: {node_mask}")
        assert False, "Variables not masked properly."


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_E_mask(X_mask: torch.Tensor, num_E_features: int | None = None) -> torch.Tensor:
    """
    Returns a mask for the edges based on the node mask.
    The edge mask is True if both nodes of the edge are present in the node mask.
    Args:
        X_mask (torch.Tensor): The node mask of shape (bs, n).
        num_E_features (int): The number of edge features. Default is 1.
        num_E_features (int | None): If provided, the edge mask will be expanded to this number of features.
    Returns:
        torch.Tensor: The edge mask of shape (bs, n, n, num_E_features) or (bs, n, n) if num_E_features is None.
    """
    assert X_mask.ndim == 2, "X_mask should be a 2D tensor"
    bs, n = X_mask.shape
    e_mask_1 = X_mask.unsqueeze(2)  # bs, n, 1
    e_mask_2 = X_mask.unsqueeze(1)  # bs, 1, n
    E_mask = e_mask_1 * e_mask_2  # bs, n, n
    # set diagonal to 0
    E_mask &= ~torch.eye(n, device=X_mask.device, dtype=torch.bool).unsqueeze(0)
    if num_E_features is None:
        return E_mask
    return expand_tensor(E_mask, (bs, n, n, num_E_features))


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2

            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def get_model(args, node_feats, edge_feats):
    from project.reconstructor.graph_transformer.transformer import GraphTransformer

    if args.small_model == 1:
        hidden_dims = {
            "dx": 16,
            "de": 8,
            "dy": 8,
            "n_head": 2,
            "dim_ffX": 16,
            "dim_ffE": 8,
            "dim_ffy": 8,
        }
        hidden_mlp_dims = {"X": 32, "E": 16, "y": 16}
    elif args.task == "abstract":
        hidden_dims = {
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 256,
        }
        hidden_mlp_dims = {"X": 128, "E": 64, "y": 128}
    else:
        # old
        # hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        # hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
        hidden_dims = {
            "dx": 128,
            "de": 64,
            "dy": 128,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 256,
        }
        hidden_mlp_dims = {"X": 256, "E": 128, "y": 128}

    model = GraphTransformer(
        input_dims={"X": node_feats, "E": edge_feats, "y": 1},
        # input_dims={'X': node_feats + 3, 'E': edge_feats, 'y': 1 + 6},
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims={"X": node_feats, "E": edge_feats, "y": 1},
        # output_dims={'X': node_feats - 1, 'E': edge_feats, 'y': 1},
        n_layers=args.num_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )

    return model
