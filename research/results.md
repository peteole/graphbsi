
## moses

| Sampler | NFE | n | Val. | Uniq. | Nov. | Filters | FCD | SNN | Scaf. |
|---|---|---|---|---|---|---|---|---|---|
| nfe=50,method=em,gamma=20,rho=0.7,bg=0.3,ms=0.5,ms_clip=False,seed=0 | 50 | 25000 | 99.6 | 100.0 | 95.5 | 98.1 | 0.682 | 0.509 | 16.0 |
| nfe=50,method=em,gamma=20,rho=0.7,bg=0.3,ms=0.5,ms_clip=False,seed=1 | 50 | 25000 | 99.5 | 100.0 | 95.8 | 98.1 | 0.687 | 0.510 | 14.3 |
| nfe=50,method=em,gamma=20,rho=0.7,bg=0.3,ms=0.5,ms_clip=False,seed=2 | 50 | 25000 | 99.6 | 100.0 | 95.7 | 98.0 | 0.696 | 0.509 | 16.8 |
| nfe=50,method=em,gamma=20,rho=0.7,bg=1.0 | 50 | 25000 | 99.6 | 99.9 | 95.0 | 98.2 | 0.671 | 0.517 | 14.9 |
| nfe=50,method=ou,gamma=20,ms=1.0 | 50 | 25000 | 99.7 | 100.0 | 94.3 | 98.6 | 0.863 | 0.526 | 15.6 |
| nfe=500,method=em,gamma=100,ms=1.5,ms_clip=False | 500 | 25000 | 99.6 | 100.0 | 94.3 | 99.1 | 0.678 | 0.525 | 17.9 |
| nfe=500,method=ou,gamma=90,ms=1.0,ms_clip=False | 500 | 25000 | 99.8 | 100.0 | 91.5 | 99.3 | 0.733 | 0.544 | 12.4 |
| *paper em_50* | 50 | 25000 | 99.3 | 100.0 | 96.5 | 96.9 | 1.060 | 0.500 | 15.2 |
| *paper ou_50* | 50 | 25000 | 99.7 | 100.0 | 94.6 | 98.2 | 1.190 | 0.520 | 15.1 |
| *paper em_500* | 500 | 25000 | 99.8 | 100.0 | 92.5 | 99.1 | 0.720 | 0.540 | 14.3 |
| *paper ou_500* | 500 | 25000 | 99.9 | 100.0 | 90.7 | 99.2 | 0.900 | 0.550 | 12.7 |

Best published, for reference:

| Metric | Best atom-level graph model | Best any model |
|---|---|---|
| Val. ↑ | 99.9 (GraphBSI) | 100.0 (JT-VAE) |
| Uniq. ↑ | 100.0 (several) | 100.0 (several) |
| Nov. ↑ | 100.0 (DMol) | 100.0 (DMol) |
| Filters ↑ | 99.2 (GraphBSI) | 99.7 (VAE) |
| FCD ↓ | 0.720 (GraphBSI) | 0.520 (CharRNN (SMILES)) |
| SNN ↑ | 0.590 (GraphBFN) | 0.590 (GraphBFN) |
| Scaf. ↑ | 23.5 (DeFoG@50) | 23.5 (DeFoG@50) |

## guacamol

| Sampler | NFE | n | Val. | V.U. | V.U.N. | KL | FCD |
|---|---|---|---|---|---|---|---|
| nfe=50,method=em,gamma=10,ms=1.0,batch_size=512 | 50 | 15000 | 98.6 | 98.6 | 98.3 | 91.0 | 71.4 |
| nfe=50,method=ou,gamma=20,ms=1.0,rho=0.7,batch_size=512 | 50 | 15000 | 99.4 | 99.4 | 98.9 | 93.7 | 81.1 |
| nfe=50,method=ou,gamma=20,ms=1.0,rho=0.7,batch_size=512,seed=1 | 50 | 15000 | 99.5 | 99.5 | 99.0 | 93.9 | 81.2 |
| nfe=50,method=ou,gamma=20,ms=1.0,rho=0.7,batch_size=512,seed=2 | 50 | 15000 | 99.4 | 99.4 | 98.8 | 93.7 | 81.0 |
| nfe=50,method=ou,gamma=40,ms=1.5,ms_clip=False,batch_size=512 | 50 | 15000 | 99.3 | 99.3 | 98.9 | 93.2 | 82.2 |
| nfe=500,method=em,gamma=200,ms=1.0,batch_size=512 | 500 | 15000 | 98.9 | 98.8 | 98.4 | 92.5 | 78.1 |
| nfe=500,method=ou,gamma=250,ms=0.0,batch_size=512 | 500 | 15000 | 99.8 | 99.8 | 98.2 | 98.4 | 80.1 |
| nfe=500,method=ou,gamma=250,ms=1.0,rho=0.7,batch_size=512 | 500 | 15000 | 99.6 | 99.6 | 98.5 | 97.1 | 80.4 |
| *paper em_50* | 50 | 15000 | 97.5 | 97.5 | 97.2 | 90.7 | 65.6 |
| *paper ou_50* | 50 | 15000 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| *paper em_500* | 500 | 15000 | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |
| *paper ou_500* | 500 | 15000 | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |

Best published, for reference:

| Metric | Best atom-level graph model | Best any model |
|---|---|---|
| Val. ↑ | 99.7 (FragFM) | 100.0 (Graph MCTS) |
| V.U. ↑ | 99.6 (GraphBSI) | 100.0 (G2PT) |
| V.U.N. ↑ | 98.7 (GraphBSI) | 98.7 (GraphBSI) |
| KL ↑ | 98.4 (GraphBSI) | 99.6 (GVT) |
| FCD ↑ | 82.6 (GraphBSI) | 93.4 (G2PT-base (85M)) |

## Pipeline reruns (`scripts/reproduce.py --tables new`)

The tables above are harness runs (`research/harness.py`, batch 256 for Moses
and 512 for GuacaMol, seeds 0/1/2). The six rows registered in
`scripts/reproduce.py` rerun the same recipes through the reproduction pipeline
instead: the released `train.py`, the paper row's own seed, sampling batch 125,
one A100 each, 0.6-3.1 h per row. Their numbers are the ones in
`results/metrics.md`.

| Row | FCD | harness FCD | Scaf. | harness Scaf. |
|---|---|---|---|---|
| moses:em_50_corrected | 0.659 | 0.688 ± 0.007 | 15.3 | 15.7 |
| moses:ou_50_corrected | 0.882 | 0.863 | 14.8 | 15.6 |
| moses:em_500_corrected | 0.669 | 0.678 | 16.3 | 17.9 |
| moses:ou_500_corrected | 0.804 | 0.733 | 13.8 | 12.4 |
| guacamol:em_50_corrected | 69.3 | 71.4 | - | - |
| guacamol:ou_50_corrected | 80.9 | 81.1 | - | - |

The spread is draw noise: seed, sampling batch size and GPU all change which
molecules come out, and the earlier seed sweep put Moses FCD at ±0.007 for an
identical configuration on an identical GPU but ±0.03 across GPUs and batch
sizes. Moses OU 500 is the widest gap (0.804 vs 0.733) and the only row where
the two measurements disagree by more than that; the pipeline number is the one
quoted everywhere, being the conservative one.

The GuacaMol Euler 50 rows are not comparable: the harness run used
`guacamol/last.ckpt`, while the paper evaluated that row on
`guacamol/epoch_59.ckpt`, which is what the pipeline row uses. 69.3 against the
paper's 65.6 is therefore the matched comparison.
