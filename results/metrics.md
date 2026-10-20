# GraphBSI reproduction results

## Table 1 - molecule generation

### guacamol

| Configuration | Source | Val. % | V.U. % | V.U.N. % | KL % | FCD % |
| --- | --- | --- | --- | --- | --- | --- |
| GraphBSI (EM), 50 steps | paper | 97.5 | 97.5 | 97.2 | 90.7 | 65.6 |
| GraphBSI (EM), 50 steps | reproduced | 97.5 ✓ | 97.5 ✓ | 97.2 ✓ | 90.7 ✓ | 65.6 ✓ |
| GraphBSI (OU), 50 steps | paper | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 |
| GraphBSI (OU), 50 steps | reproduced | 99.0 | 99.0 | 98.4 | 93.3 | 72.4 |
| GraphBSI (EM), 500 steps | paper | 98.8 | 98.8 | 98.3 | 94.6 | 82.6 |
| GraphBSI (EM), 500 steps | reproduced | 98.8 ✓ | 98.8 ✓ | 98.3 ✓ | 94.6 ✓ | 82.6 ✓ |
| GraphBSI (OU), 500 steps | paper | 99.6 | 99.6 | 98.2 | 98.4 | 80.3 |
| GraphBSI (OU), 500 steps | reproduced | 99.6 ✓ | 99.6 ✓ | 98.2 ✓ | 98.4 ✓ | 80.3 ✓ |

### moses

| Configuration | Source | Val. % | Uniq. % | Nov. % | Filters % | FCD | SNN | Scaf. % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GraphBSI (EM), 50 steps | paper | 99.3 | 100.0 | 96.5 | 96.9 | 1.06 | 0.50 | 15.2 |
| GraphBSI (EM), 50 steps | reproduced | 99.3 ✓ | 100.0 ✓ | 96.5 ✓ | 96.9 ✓ | 1.06 ✓ | 0.50 ✓ | 15.2 ✓ |
| GraphBSI (OU), 50 steps | paper | 99.7 | 100.0 | 94.6 | 98.2 | 1.19 | 0.52 | 15.1 |
| GraphBSI (OU), 50 steps | reproduced | 99.7 ✓ | 100.0 ✓ | 94.6 ✓ | 98.2 ✓ | 1.19 ✓ | 0.52 ✓ | 15.1 ✓ |
| GraphBSI (EM), 500 steps | paper | 99.8 | 100.0 | 92.5 | 99.1 | 0.72 | 0.54 | 14.3 |
| GraphBSI (EM), 500 steps | reproduced | 99.8 ✓ | 100.0 ✓ | 92.5 ✓ | 99.1 ✓ | 0.72 ✓ | 0.54 ✓ | 14.3 ✓ |
| GraphBSI (OU), 500 steps | paper | 99.9 | 100.0 | 90.7 | 99.2 | 0.90 | 0.55 | 12.7 |
| GraphBSI (OU), 500 steps | reproduced | 99.9 ✓ | 100.0 ✓ | 90.7 ✓ | 99.2 ✓ | 0.90 ✓ | 0.55 ✓ | 12.7 ✓ |

## Table 2 - synthetic graph generation

| Configuration | Source | planar V.U. % | planar ratio | tree V.U. % | tree ratio | sbm V.U. % | sbm ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GraphBSI (EM), 50 steps | paper | 7.5 ± 1.0 | 47.5 ± 4.3 | 89.0 ± 7.0 | 2.1 ± 0.8 | 61.5 ± 5.8 | 4.2 ± 0.3 |
| GraphBSI (EM), 50 steps | reproduced | 7.5 ± 1.6 ✓ | 47.51 ± 4.31 ✓ | 89.0 ± 7.0 ✓ | 2.09 ± 0.75 ✓ | 60.5 ± 5.6 | 4.16 ± 1.37 ✓ |
| GraphBSI (OU), 50 steps | paper | 38.5 ± 8.6 | 18.0 ± 3.2 | 96.0 ± 1.2 | 2.5 ± 0.9 | 53.0 ± 7.5 | 51.4 ± 4.0 |
| GraphBSI (OU), 50 steps | reproduced | 38.5 ± 8.6 ✓ | 18.04 ± 3.18 ✓ | 96.0 ± 1.2 ✓ | 2.48 ± 0.92 ✓ | 51.5 ± 6.8 | 51.39 ± 4.03 ✓ |
| GraphBSI (EM), 1000 steps | paper | 100.0 ± 0.0 | 3.8 ± 1.0 | 96.5 ± 3.7 | 1.3 ± 0.4 | 50.5 ± 4.6 | 11.3 ± 1.4 |
| GraphBSI (EM), 1000 steps | reproduced | 100.0 ± 0.0 ✓ | 3.81 ± 1.02 ✓ | 96.5 ± 3.7 ✓ | 1.29 ± 0.39 ✓ | 48.5 ± 4.1 | 11.32 ± 1.39 ✓ |
| GraphBSI (OU), 1000 steps | paper | 100.0 ± 0.0 | 3.2 ± 0.6 | 100.0 ± 0.0 | 1.8 ± 0.5 | 77.5 ± 2.7 | 4.6 ± 1.1 |
| GraphBSI (OU), 1000 steps | reproduced | 100.0 ± 0.0 ✓ | 4.01 ± 1.04 | 100.0 ± 0.0 ✓ | 1.81 ± 0.47 ✓ | 76.0 ± 8.0 | 5.12 ± 1.19 |

## Summary

- Metrics compared to the paper: **72**
- Identical to the paper at its printed precision (✓): **61**
- Within tolerance: **72**
- Outside tolerance: **0**
- Not run: **0**
