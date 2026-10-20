# Reproduction of Tables 1 and 2

`results/` holds the output of `bash reproduce_all.sh` (Tabs. 1 and 2) run from
a fresh clone with the released checkpoints. `results/metrics.md` puts every
reproduced metric next to the published one; ✓ marks values that are identical
to the paper at its printed precision.

## Result

All 20 rows of Tabs. 1 and 2 were rerun. All 72 compared metrics are within
tolerance and 61 are identical to the paper at its printed precision.

- **Identical:** every Moses row, GuacaMol EM 50, EM 500 and OU 500, and all
  planar and tree cells except the planar OU-1000 ratio. For these molecule rows
  every logged metric equals the paper run's value at full precision (Moses EM
  500 differs in the 16th significant digit of the KL score).
- **SBM:** the three replayable rows have identical ratios, but their V.U. is
  1-2 points from the paper because the SBM validity test is not seeded (see
  Notes).
- **Not replayable:** GuacaMol OU 50, planar OU 1000 and SBM OU 1000 (see
  Notes) are within tolerance but not identical.

## How the paper numbers are replayed

Sampling is deterministic given the seed, the sampling batch size, the software
environment and the GPU architecture. Every row in `scripts/reproduce.py`
therefore records the paper run it comes from and replays it:

- **Seed.** Each row uses the seed of its W&B evaluation run (`seed`,
  `source`).
- **Environment.** Moses, GuacaMol, planar and tree were evaluated in the uv
  environment (`uv.lock`, PyTorch 2.7.0), SBM in the Pixi environment
  (`pixi.lock`, PyTorch 2.8.0, graph-tool). `reproduce.py` runs each row with the
  interpreter of its environment.
- **Batch size.** Molecules are sampled in batches of 64; synthetic graphs are
  scored on 200 samples in 5 batches of 40.
- **Caches.** Dataset and benchmark caches are built in a separate process
  before any row runs. Building them inside a seeded run instantiates the FCD
  network and shifts the random state, so the samples no longer match. The
  paper runs of the two 500-step Moses rows did build it after seeding, so
  those rows rebuild it in the run (`plots.0.rebuild_benchmarker`).
- **GPU.** The row's `gpu` field names the GPU of the paper run, and
  `reproduce.py` prints a note when the current GPU differs. Matching the GPU
  architecture is needed for bit-identical samples.
- **Checkpoint.** The GuacaMol EM-50 row was evaluated while the training run
  was at epoch 59 (`guacamol/epoch_59.ckpt`); all other rows use `last.ckpt`.

## Rows

| Table | Row | Paper run | GPU | Replayable |
|-------|-----|-----------|-----|------------|
| 1 | Moses EM/OU 50, EM/OU 500 | `molecules_ablation_12_layers_moses/{7zrha335,6mov362j,7q53x74d,mlsggyxe}` | GTX 1080 Ti | yes |
| 1 | GuacaMol EM 50, EM/OU 500 | `molecules_ablation_12_layers_guacamol/{o9pvyeps,y264r6rd,ejp3imis}` | A100 | yes |
| 1 | GuacaMol OU 50 | `molecules_ablation_12_layers_guacamol/sp33mijc` | A100 | no (see below) |
| 2 | Planar EM 50, OU 50, EM 1000 | `ablations_planar/{8q2dpcx6,bti8kpmr,enf9hwf3}` | GTX 1080 Ti | yes |
| 2 | Tree EM 50, OU 50 | `ablations_tree/{lvjkewm6,q9wb7nvs}` | GTX 1080 Ti | yes |
| 2 | Tree EM 1000, OU 1000 | `ablations_tree/{w35zc7cq,ut4wllb4}` | A100 | yes |
| 2 | SBM EM 50, OU 50, EM 1000 | `ablations_sbm/{qdmayuv3,hij7alnp,mmy3zxio}` | RTX 2080 Ti | yes |
| 2 | Planar OU 1000, SBM OU 1000 | validation during training (`train_planar/agag753c`, `train_sbm/aog3lu3i`) | - | no |

## Notes

- **SBM V.U.** The SBM validity test of `synthetic-graph-benchmarks` runs a
  graph-tool fit that is not seeded, so SBM V.U. can differ slightly between two
  scorings of the same graphs.
- **Rows that cannot be replayed exactly.** GuacaMol OU 50 was evaluated on an
  epoch-69 checkpoint that training later overwrote, so the row uses the paper
  run's seed on `guacamol/last.ckpt`. Planar and SBM OU 1000 were evaluated
  during training and have no evaluation seed; these two rows run with seed 0.
  The three rows reproduce the paper within tolerance but not digit for digit.
