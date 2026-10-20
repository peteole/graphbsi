#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings
from multiprocessing import set_start_method
from typing import Optional

import hydra
import lightning.pytorch as pl
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict

from project.class_conditioner.class_conditioner import ClassConditioner
from project.plots.kl_plots_callback import KLPlotsCallback
from project.tasks import ELBO
from project.utils import (
    filter_device_available,
    get_logger,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)
from project.utils.callbacks import (
    AlwaysSaveCheckpoints,
    ConfigInCheckpoint,
    WandbSummaries,
)
from project.utils.metrics_capture import MetricsCaptureCallback, flush, install_wandb_hook

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

install_wandb_hook()

# If data loading is really not a bottleneck for you, uncomment this to silence the
# warning about it
# warnings.filterwarnings(
#     "ignore",
#     "The '\w+_dataloader' does not have many workers",
#     module="lightning",
# )
warnings.filterwarnings(
    "ignore",
    "The `srun` command is available on your system but is not used",
    module="lightning",
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    filter_device_available
)


log = get_logger()


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


def get_callbacks(config, logger, datamodule):
    monitor = {"monitor": "val/ELBO", "mode": "max"}
    callbacks = [
        KLPlotsCallback(datamodule=datamodule),
        WandbSummaries(**monitor),
        ModelCheckpoint(
            dirpath=f"checkpoints/{logger.name}/{logger.version}",
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
        LearningRateMonitor(),
        AlwaysSaveCheckpoints(),
        ConfigInCheckpoint(config),
    ]
    if "plots" in config:
        for cb_config in config.plots:
            plots_cb = instantiate(cb_config, datamodule=datamodule)
            callbacks.append(plots_cb)
    if config.early_stopping is not None:
        stopper = EarlyStopping(
            patience=int(config.early_stopping),
            min_delta=0,
            strict=False,
            check_on_train_epoch_end=False,
            **monitor,
        )
        callbacks.append(stopper)
    # Install last so on_validation_epoch_end flush runs after plot callbacks.
    callbacks.append(MetricsCaptureCallback())
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    rng = set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)
    model_checkpoint_path = config.get("load_checkpoint", None)
    resume_from_epoch = config.get("resume_from_epoch", None)
    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Loading data")
    datamodule: pl.LightningDataModule = instantiate(config.data, rng=rng)
    datamodule.prepare_data()
    # Force spawn only after CPU-side dataset preprocessing (ProcessPool).
    set_start_method("spawn", force=True)
    datamodule.setup("fit")

    log.info("Instantiating model")
    reconstructor = instantiate(config.reconstructor, datamodule)
    # reconstructor = torch.compile(reconstructor)
    belief = instantiate(config.belief, datamodule=datamodule)

    class_conditioner: Optional[ClassConditioner] = None
    if "class_conditioner" in config:
        class_conditioner = instantiate(config.class_conditioner, datamodule=datamodule)
    task: ELBO = instantiate(
        config.task,
        reconstructor=reconstructor,
        belief=belief,
        class_conditioner=class_conditioner,
        datamodule=datamodule,
    )

    if model_checkpoint_path is not None:
        log.info(f"Inference mode, loading model from checkpoint {model_checkpoint_path}")
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {model_checkpoint_path} does not exist.")
        if resume_from_epoch is None:
            config.wandb.name = f"inference_{config.wandb.name}_samplers={",".join([s.name for s in task.samplers])}"
            task.skip_validation = True
    logger = instantiate(
        config.wandb,
        _target_="lightning.pytorch.loggers.WandbLogger",
        resume=(config.wandb.mode == "online") and "allow",
        log_model=True,
    )

    log_hyperparameters(logger, config, reconstructor)
    install_wandb_hook()

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, logger, datamodule)
    plugins = []
    if os.environ.get("SLURM_JOB_ID") is not None:
        plugins.append(SLURMEnvironment(auto_requeue=False))
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
    )
    if model_checkpoint_path is not None:
        if resume_from_epoch is None:
            trainer.fit_loop.epoch_progress.current.completed = 100# dummy value to avoid callbacks being skipped in first epoch
            install_wandb_hook()
            trainer.validate(task, datamodule=datamodule, ckpt_path=model_checkpoint_path)
            print("Done validating model from checkpoint")
            flush()
            return
        else:
            resume_from_epoch = int(resume_from_epoch)
            log.info(f"Resuming training from epoch {resume_from_epoch}")
            trainer.fit_loop.epoch_progress.current.completed = resume_from_epoch
            trainer.fit(task, datamodule=datamodule, ckpt_path=model_checkpoint_path)
    else:
        log.info("Starting training!")
        trainer.fit(task, datamodule=datamodule)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()
    flush()
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    best_score = trainer.checkpoint_callback.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
