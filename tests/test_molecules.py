from typing import Optional

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from project.class_conditioner.class_conditioner import ClassConditioner
from project.data.molecule import MoleculeDataModule
from project.plots.molecule import is_valid_molecule
from project.reconstructor.graph_transformer.mock_graph_reconstructor import (
    MockMoleculeReconstructor,
)
from project.samplers.sampler import Sampler
from project.utils.seed import set_seed
from project.utils.tensor_utils import dict_of_tensors_to_tensor_dict

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def test_molecule_scores(tmp_path):
    """Sampling with a reconstructor that returns training molecules must
    produce exactly those (valid) molecules."""
    GlobalHydra.instance().clear()
    initialize(config_path="../config", job_name="test", version_base=None)
    config = compose(
        config_name="train",
        overrides=[
            "experiment=qm9/train",
            f"data.cache_dir={tmp_path}",
            "+data.max_num_samples=100",
            "task.sampler.variable_noise.max_function_evals=20",
        ],
    )
    rng = set_seed(config)
    molecular_data: MoleculeDataModule = instantiate(config.data, rng=rng)
    molecular_data.prepare_data()
    molecular_data.setup("fit")
    samples = next(iter(molecular_data.train_dataloader()))
    samples = dict_of_tensors_to_tensor_dict(samples)
    samples = samples.to(device)
    reconstructor = MockMoleculeReconstructor(samples)
    belief = instantiate(config.belief, datamodule=molecular_data)
    belief.to(device)
    class_conditioner: Optional[ClassConditioner] = None
    if "class_conditioner" in config:
        class_conditioner = instantiate(
            config.class_conditioner, datamodule=molecular_data
        )

    sampler: Sampler = instantiate(
        config.task.sampler.variable_noise,
        belief=belief,
        reconstructor=reconstructor,
        class_conditioner=class_conditioner,
    )
    sampler.to(device)
    generated_samples, _ = sampler.sample(len(samples), len(samples))
    mols = []
    for i in range(generated_samples.batch_size[0]):
        mol = molecular_data.molecular_sample_to_molecule(
            {
                "X": generated_samples["X"][i],
                "E": generated_samples["E"][i],
                "mask": generated_samples["mask"][i],
            }
        )
        assert mol is not None, "Failed to create molecule"
        assert is_valid_molecule(mol), "Invalid molecule generated"
        mols.append(mol)
