import networkx as nx
import torch

from project.belief.belief_constructor import construct_synthetic_graph_belief_from_config
from project.data.synthetic_graphs import graph_to_sample
from project.reconstructor.graph_transformer.edge_features import RRWPEdgeFeatures
from project.scheduler.ole_scheduler import OLEScheduler
from project.utils import tensor_utils

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def test_rrwp_features():
    num_nodes = 8
    num_graphs = 5
    rrwp_features = RRWPEdgeFeatures(max_steps=num_nodes, make_symmetric=True)
    belief = construct_synthetic_graph_belief_from_config(
        max_num_nodes=num_nodes,
        datamodule=None,  # only needed for marginal_as_prior
        device=str(device),
        initial_categorical_belief_std=1.0,
        scheduler=OLEScheduler(5.0, 20.0),
    )
    sbm_graphs = [
        nx.stochastic_block_model(
            sizes=[num_nodes // 2, num_nodes // 2],
            p=[[0.8, 0.2], [0.2, 0.8]],
            seed=seed,
        )
        for seed in range(num_graphs)
    ]
    ground_truths = [graph_to_sample(g, num_nodes) for g in sbm_graphs]
    ground_truths = {
        k: torch.stack([g[k] for g in ground_truths]) for k in ground_truths[0].keys()
    }
    ground_truths = tensor_utils.dict_of_tensors_to_tensor_dict(ground_truths).to(device)
    b = belief.sample_belief_at_t(
        ground_truths, t=torch.full(ground_truths.batch_size, 0.3, device=device)
    )
    features = rrwp_features(b)
    assert features.shape == (num_graphs, num_nodes, num_nodes, num_nodes)
    # Step 0 is the identity and every step is a row-stochastic matrix power.
    eye = torch.eye(num_nodes, device=device).expand(num_graphs, -1, -1)
    assert torch.allclose(features[..., 0], eye)
    assert torch.allclose(
        features.sum(dim=2), torch.ones_like(features.sum(dim=2)), atol=1e-4
    )
