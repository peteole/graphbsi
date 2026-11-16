# Best published values on MOSES / GuacaMol / QM9 — target-setting report for GraphBSI

Compiled 2026-10-22. Every number is quoted **exactly as printed** in the cited source. Where I could
not verify a number from a primary source (LaTeX/PDF of the paper), it is marked *(unverified)*.

Primary-source method: arXiv LaTeX sources (`https://arxiv.org/e-print/<ID>`) and `pdftotext -layout`
on publisher PDFs, so numbers are read off the actual table markup rather than paraphrased.

---

## 0. READ THIS FIRST — three protocol traps that invalidate naive comparisons

If you read only one thing: **GraphBSI arXiv v1 and v2 report different MOSES numbers, and v2 is the
correct one** (Trap 2). The MOSES FCD to beat is **0.72** (GraphBSI v2 EM@500) / **0.58** (FragFM), not
0.26; the Scaf to beat is **23.5** (DeFoG@50), and GraphBSI sits at 15.7.

### Trap 1 — MOSES FCD / SNN / Scaf are reported on **TestSF** (scaffold split) in the DiGress lineage

DiGress states it explicitly (arXiv:2209.14734, §"Molecule generation at scale"):

> "For MOSES, the reported scores for FCD, SNN, and Scaffold similarity are computed on the dataset
> made of separate scaffolds, which measures the ability of the networks to predict new ring structures."

Cometh states it independently too (arXiv:2406.06449, appendix on large datasets):

> "The MOSES benchmark also computes metrics by comparing the generated set to a scaffold test set,
> from which we report the FCD, SNN … and Scaffold similarity (Scaf)."

The convention is inherited by **DisCo, Cometh, DeFoG, GraphBFN, Pard, AutoGraph, DMol, MolHIT,
FragFM, LG-Flow, Flowette, GraphBSI v2** (DisCo: "consistent with DiGress … official MOSES evaluation
code"; AutoGraph: "following the data splits and experimental setup from DiGress"; GraphBFN:
"evaluation metrics … of MOSES follow from Vignac et al."). Cross-check against the official MOSES
leaderboard, which reports both splits:

| Reference model | FCD/Test | FCD/TestSF | SNN/Test | SNN/TestSF | Scaf/Test | Scaf/TestSF |
|---|---|---|---|---|---|---|
| MOSES `Train` row | 0.008 | 0.4755 | 0.6419 | 0.5859 | 0.9907 | 0.0 |
| VAE (SMILES) | 0.099 | **0.567** | 0.6257 | **0.5783** | 0.9386 | **0.0588** |
| CharRNN | 0.0732 | **0.5204** | 0.6015 | **0.5649** | 0.9242 | **0.1101** |

DiGress/DeFoG print **VAE = FCD 0.57, SNN 0.58, Scaf 5.9** → these are exactly the **TestSF** column.
Confirmed. Likewise MolHIT and FragFM print a `Training set` row with **FCD 0.48** (= 0.4755, TestSF)
and MolHIT prints **CharRNN FCD 0.52** (= 0.5204, TestSF).

**Useful floor:** on TestSF, the training set itself scores FCD/TestSF = 0.4755. A model trained on the
MOSES train split essentially cannot go below ~0.48. **Any reported MOSES "FCD" below ~0.48 is
therefore not TestSF** — it is Test or validation. Similarly **Scaf ≈ 0.88–0.99 is Test, Scaf ≈ 0.02–0.24
is TestSF** (ceiling on TestSF for real models is ≈ 0.11 for SMILES models, ≈ 0.24 for DeFoG@50).

### Trap 2 — GraphBSI v1 vs v2: the MOSES FCD/SNN/Scaf numbers were **silently corrected**

arXiv:2511.03015 has two versions, and **the MOSES columns differ**. v1 reported Test-split numbers
against TestSF baselines; v2 recomputed them on TestSF. **Use v2.** (v2 is the latest; there is no v3.
The OpenReview PDF for `py2NPbAdvH` sits behind a browser-verification wall and could not be fetched.)

| GraphBSI config | ver | Moses FCD↓ | SNN↑ | Scaf↑ | Nov.↑ | Val.↑ | Filters↑ |
|---|---|---|---|---|---|---|---|
| EM, 50 steps | **v1** | 0.50 | 0.53 | 86.2 | 96.2 | 99.4 | 97.3 |
| EM, 50 steps | **v2** | 1.06 | 0.50 | 15.2 | 96.5 | 99.3 | 96.9 |
| OU, 50 steps | **v1** | 0.60 | 0.55 | 88.0 | 94.3 | 99.7 | 98.5 |
| OU, 50 steps | **v2** | 1.19 | 0.52 | 15.1 | 94.6 | 99.7 | 98.2 |
| EM, 500 steps | **v1** | 0.26 | 0.57 | 86.8 | 92.3 | 99.8 | 99.1 |
| EM, 500 steps | **v2** | **0.72** | 0.54 | 14.3 | 92.5 | 99.8 | 99.1 |
| OU, 500 steps | **v1** | 0.41 | 0.59 | 88.2 | 90.3 | 99.9 | 99.3 |
| OU, 500 steps | **v2** | 0.90 | 0.55 | 12.7 | 90.7 | 99.9 | 99.2 |

GuacaMol numbers are **identical in v1 and v2**. QM9 changed only in the `Train Set` validity row
(100.0 → 99.3).

Note: the v2 metric-description table (Table 7) was *not* updated — it still says FCD is computed
against "the validation set" and SNN/Scaf against "the test set", which contradicts the corrected
numbers. Treat that table text as stale.

### Trap 3 — transcription errors, re-runs and reporting inconsistencies across the published tables

* **GraphBSI (v1 and v2) lists GraphBFN MOSES Validity = 98.5.** The GraphBFN paper (PMLR v267, Table 3)
  prints **88.5**. Verified byte-exact from the PMLR PDF. GraphBSI's claim of "pushing FCD from 1.07"
  is correct, but GraphBFN is *not* a 98.5-validity model.
* **G2PT lists DeFoG MOSES Filters = 99.9 and GuacaMol KL = 97.9.** DeFoG's own paper says **98.9** and
  **97.7**.
* **Cometh:** papers quoting "Cometh" sometimes mean **Cometh-PC** (the predictor-corrector variant).
  Cometh's own paper: Cometh MOSES Val 87.0, Cometh-PC 90.5; GuacaMol Cometh Val 94.4, Cometh-PC 98.9.
  DeFoG/GraphBSI/MolHIT quote the **-PC** numbers under the plain name "Cometh".
  Also, Cometh's own MOSES table has **no Novelty column** (it reports Val, Val&Uni, VUN); the
  "Cometh Novelty = 92.6" that circulates in DeFoG/GraphBSI/Flowette tables is not in Cometh's main table.
* **MolHIT re-ran the baselines itself** and gets different numbers from the originals
  (DiGress Val 87.1 / FCD 1.25 / Scaf 12.8 vs published 85.7 / 1.19 / 14.8; Cometh 87.2 / 1.44 / 16.8
  vs 87.0 / 1.44 / 15.9). Its GuacaMol table uses the **unfiltered** GuacaMol dataset → not comparable.
  MolHIT's MOSES `Training set` row is FCD 0.48 / SNN 0.59 (correct TestSF reference values), unlike the
  DeFoG/GraphBSI lineage which prints 0.01 / 0.64 (Test values) — so MolHIT's table is internally
  consistent where the others are not.
* **DMol also re-ran every baseline** "on the same version of MOSES" and gets yet another set:
  DiGress 84.8/100/94.5/97.2/1.18/0.55/14.6, DisCo 85.7/100/97.4/95.8/1.40/0.51/14.5,
  DeFoG@50 84.2/100/97.2/96.9/1.89/0.50/14.8. GuacaMol re-runs: DiGress 84.5/100/99.8/93.1/68.4,
  DisCo 86.0/100/99.8/92.8/59.7, DeFoG 86.7/100/99.5/92.5/58.1. Treat re-run columns as a separate
  protocol.
* **The GuacaMol FCD score is sensitive to the `fcd` package version.** AutoGraph reports that "using
  the latest FCD package gives a FCD score of **94.7** for the pre-trained AutoGraph" instead of the
  **91.4** it tabulates. Any GuacaMol FCD comparison across papers from different years carries this
  uncertainty — worth stating if you claim a record.
* **AutoGraph / GEEL / GVT compute their metrics on SMILES**, not on graphs. AutoGraph notes ~20% of
  GuacaMol molecules do not round-trip SMILES→graph→SMILES, which is exactly the filter DiGress applies.
* **DisCo's reproduction of the VAE row has a typo**: `Unique 98.8` where the MOSES benchmark and every
  other paper print 99.8.
* **AutoGraph's GuacaMol baseline row for MCTS is wrong**: it prints Novelty 99.4 / KL 52.2 where
  DiGress/DeFoG/Cometh print 95.4 / 82.2.
* **GuacaMol filtering:** all DiGress-lineage graph models train on a *filtered* GuacaMol
  (SMILES→graph→SMILES round-trip filter from DiGress); SMILES models use the full 1.27M set.
* **GuacaMol V.U./V.U.N. vs Uniqueness/Novelty:** DiGress reports LSTM as Unique 100 / Novel 91.2;
  DeFoG recomputes the same model as V.U. 95.9 / V.U.N. 87.4. Not the same quantity.

---

## 1. Best published value per metric

### 1a. MOSES — DiGress-lineage protocol (FCD/SNN/Scaf on **TestSF**, 25k or 10k samples)

This is the protocol GraphBSI competes in, so these are the numbers to beat.

| Metric | Best overall (any model type) | Method | Venue / year | Best **graph-based one-shot** (diffusion/flow) | Method |
|---|---|---|---|---|---|
| Validity ↑ | **100.0** | JT-VAE (fragment), MolGrow (one-shot) | NeurIPS'18 / ICML'21 | **99.9** | GraphBSI (OU, 500) — arXiv:2511.03015v2 |
| Uniqueness ↑ | **100.0** | many (DiGress, DisCo, FragFM, G2PT, GraphBSI) | — | **100.0** | GraphBSI, DiGress, DisCo |
| Novelty ↑ | **100.0** | MolHF (Val. 71.0, Filters 17.8 — degenerate); **DMol** (Val. 87.8, Filters 97.8, FCD 1.12 — *not* degenerate) | IJCAI'23 / arXiv:2504.06312 | **100.0** | DMol — arXiv:2504.06312 |
| " (among GraphBSI-class) | 99.9 | JT-VAE | NeurIPS'18 | 99.2 / 98.1 | GraphBSI (EM, 10 steps) / Flowette |
| Filters ↑ | **99.7** | VAE (SMILES) | MOSES paper | **99.6** | GraphBSI (FB, β₀=1, 500) — but FCD 2.84 |
| " (usable config) | 99.7 | VAE | — | 99.2 | GraphBSI (OU, 500); Cometh-PC 99.1; FragFM 99.1 |
| FCD ↓ (TestSF) | **0.52** | CharRNN | MOSES paper (= FCD/TestSF 0.5204) | **0.58** | FragFM (300 steps) — arXiv:2502.15805 |
| " 2nd/3rd | 0.567 VAE · 0.71 SAFE-GPT | — | — | **0.72 GraphBSI (EM,500)** · 0.90 GraphBSI (OU,500) · 0.91 AutoGraph · 0.97 G2PT-base · 1.00 Pard | — |
| SNN ↑ (TestSF) | **0.59** | GraphBFN; GraphBSI (FB); MOSES `Train` row is 0.5859 | ICML'25 | **0.59** | GraphBFN (500) — PMLR v267; DMol 0.58 |
| Scaf ↑ (TestSF) | **23.5** | DeFoG (50 steps) | ICML'25 / arXiv:2410.04263 | **23.5** | DeFoG (50 steps) |
| " 2nd | 18.4 | Cometh @50 steps (FragFM's re-run) | — | 18.4 | Cometh @50 |

Notes on the records:
* **FCD 0.52 (CharRNN)** is the hard overall record. Among *graph* models the record is **FragFM 0.58**
  — but FragFM is **fragment-level**, not atom-level. The best **atom-level graph** FCD is
  **GraphBSI (EM, 500) = 0.72**, then AutoGraph 0.91 (AR, ≈113M), G2PT-base 0.97 (AR, 85M),
  Pard 1.00, GraphBFN 1.07, MolHIT 1.03, DMol 1.12. So GraphBSI already holds the atom-level record
  here; the gap to close is to FragFM/CharRNN.
* **SNN is saturated**: 0.59 equals the MOSES `Train` row's own SNN/TestSF (0.5859). Little headroom.
* **Scaf/TestSF is anti-correlated with everything else**: DeFoG's record 23.5 comes at Validity 83.9
  and FCD 1.87. The *training set*'s Scaf/TestSF is **0.0 by construction** (TestSF scaffolds are
  disjoint from train), so high Scaf/TestSF means generating *unseen* scaffolds. This is the one MOSES
  metric where GraphBSI is clearly behind (12.7–15.7 vs 23.5), and it is *not* a saturation artifact.
* **Validity 100.0** by JT-VAE/MolGrow is by hard-coded valency construction, not learned.

### 1b. MOSES — **Test**-split protocol (official MOSES leaderboard; not comparable to 1a)

| Metric | Best value | Method | Source |
|---|---|---|---|
| Valid ↑ | 1.0 | Combinatorial, JTN-VAE | MOSES leaderboard |
| Unique@10k ↑ | 0.9996 | JTN-VAE | MOSES leaderboard |
| Novelty ↑ | 0.9994 | HMM (Valid only 0.076 — degenerate) | MOSES leaderboard |
| Filters ↑ | 0.997 | VAE | MOSES leaderboard |
| FCD/Test ↓ | **0.0732** | CharRNN | MOSES leaderboard |
| SNN/Test ↑ | **0.6257** | VAE | MOSES leaderboard |
| Scaf/Test ↑ | **0.9386** | VAE | MOSES leaderboard |
| Scaf/TestSF ↑ | **0.1101** | CharRNN | MOSES leaderboard |
| IntDiv ↑ | 0.8738 | NGram | MOSES leaderboard |

**Caution:** GraphBSI **v1**'s MOSES FCD 0.26 / SNN 0.59 / Scaf 88.2 belong in *this* table, not 1a
(FCD 0.26 is below the TestSF floor of 0.4755 and Scaf 88.2 is far above the TestSF ceiling).
Compared against this table, v1 was **not** SOTA (CharRNN FCD/Test 0.0732 and VAE Scaf/Test 0.9386 win).
v2 fixed this.

### 1c. GuacaMol distribution-learning benchmark

FCD column = the GuacaMol **FCD score** `exp(-0.2·FCD)`, reported ×100 by graph papers and in [0,1] by
the original paper. Training-set reference: Val 100.0, KL 99.9, FCD score 92.8.

| Metric | Best overall | Method | Venue / year | Best **graph one-shot diffusion/flow** | Method |
|---|---|---|---|---|---|
| Validity ↑ | **100.0** | Graph MCTS; Random SMILES sampler | GuacaMol 2019 | **99.7** | FragFM — arXiv:2502.15805 |
| " (learned, non-trivial) | 99.7 | FragFM | ICLR'26 | 99.6 | GraphBSI (OU, 500 & γ→∞, 50) |
| V.U. ↑ | **100.0** | Graph MCTS; G2PT (all sizes) | — | **99.6** | GraphBSI (OU, 500) |
| V.U.N. ↑ | **99.8** | G2PT-small (Novelty, not V.U.N.) | arXiv:2501.01073 | **98.7** | GraphBSI (OU, 50) |
| " (true V.U.N.) | 98.7 | GraphBSI (OU, 50) | ICLR'26 | 98.7 | GraphBSI (OU, 50); 98.3 GraphBSI(EM,500); 98.0 Flowette |
| KL div score ↑ | **99.61** | GVT (Graph VQ-Transformer) | arXiv:2512.02667 (preprint) | **98.4** | GraphBSI (OU, 500) |
| " runners-up | 99.4 FO-ARM · 99.35 MolGram-87.5M · 99.1 SMILES LSTM · 99.8 Random sampler | — | — | 97.9 Flowette · 97.7 DeFoG · 97.4 FragFM & GraphBSI(γ→∞,FP) | — |
| FCD score ↑ | **93.4** | G2PT-base (85M) | arXiv:2501.01073 | **85.8** | FragFM (fragment-level, one-shot) |
| " runners-up | 92.7 G2PT-large · 91.4 AutoGraph\* (94.7 w/ newer `fcd` pkg) · 91.3 SMILES LSTM · 91.12 MolGram · 90.3 FO-ARM · 85.50 GVT | — | — | **82.6 GraphBSI (EM, 500)** (best **atom-level one-shot** graph) · 81.8 LG-Flow · 80.3 GraphBSI(OU,500) · 79.2 AutoGraph · 75.1 GraphBSI(γ→∞,FP) · 73.8 DeFoG | — |

Notes:
* **KL 99.61 (GVT) and FCD 93.4 (G2PT-base) both exceed the training-set reference (99.9 / 92.8 resp.
  — KL does not exceed it, FCD does).** G2PT-base's 93.4 > training set's 92.8 means distribution
  matching is saturated at the noise floor there; treat as "tied with the data".
* The original GuacaMol **Random SMILES sampler** scores Validity 1.000, KL 0.998, FCD 0.929 — i.e. it
  beats every learned model on KL and nearly all on FCD, while having Novelty 0.000. Any "best KL"
  claim must be read alongside novelty.
* **Best headroom for GraphBSI on GuacaMol is the FCD score**: 82.6 (atom-level graph record) vs 85.8
  (FragFM, fragments) vs 93.4 (G2PT, autoregressive sequence) vs 92.8 (training set).

### 1d. QM9 (lower priority)

| Metric | Best | Method | Source |
|---|---|---|---|
| QM9 no-H Validity ↑ | **99.9** | GraphBSI (50 and 500 steps) | arXiv:2511.03015v2 |
| QM9 no-H Uniqueness ↑ | 96.8 ± 0.2 | Cometh (500) / G2PT-base 96.8 | arXiv:2406.06449 / 2501.01073 |
| QM9 no-H FCD ↓ | **0.06** | G2PT (small/base/large) | arXiv:2501.01073 |
| " best graph diffusion/flow | **0.09** | GraphBSI (500) | arXiv:2511.03015v2 |
| " | 0.11 / 0.12 | Cometh (per G2PT's table) / DeFoG (500) | — |
| QM9 with-H Validity ↑ | **99.8** | GraphBSI (500) | arXiv:2511.03015v2 |
| QM9 with-H Uniqueness ↑ | 97.6 ± 0.4 | DiGress (500) | arXiv:2209.14734 |
| QM9 with-H FCD ↓ | **0.05 ± 0.00** | DeFoG (500) — equals the train-set value | arXiv:2410.04263 |
| QM9 no-H NSPDK ↓ | 0.0001 | SID | via Flowette table *(unverified)* |
| QM9 no-H Validity (NSPDK protocol) | 99.88 | GBD | via Flowette table *(unverified)* |

GraphBSI v2's QM9 `Train Set` row is Val 99.3 / Uniq 100.0 / FCD 0.05 (both with and without H).
QM9 is reported under two incompatible protocols (DiGress-style Val/Uniq/FCD vs GDSS-style
Val/Uniq/NSPDK) — do not mix the two tables.

---

## 2. Full table of all methods found

### 2.0 Venue index for the graph-generation methods

| Method | Paper / venue | arXiv | MOSES? | GuacaMol? |
|---|---|---|---|---|
| DiGress | ICLR 2023 | 2209.14734 | yes | yes |
| DisCo | NeurIPS 2024 | 2405.11416 | yes | yes |
| Cometh | arXiv 2024 (ICLR-2025 template; acceptance unconfirmed) | 2406.06449 | yes | yes |
| DeFoG | **ICML 2025** | 2410.04263 | yes | yes |
| GraphBFN ("Smooth Interpolation for Improved Discrete Graph Generative Models") | **ICML 2025**, PMLR v267:56363–56388 | **no arXiv ID found** | yes | no |
| Pard | NeurIPS 2024 | 2402.03687 | yes | no |
| LGD | NeurIPS 2024 | 2402.02518 | yes (partial) | no |
| GEEL ("A Simple and Scalable Representation…") | ICLR 2024 | 2312.02230 | yes — **third protocol** | no |
| G2PT ("Graph Generative Pre-trained Transformer") | 2025 | 2501.01073 | yes | yes |
| FragFM | ICLR 2026 (template) | 2502.15805 | yes | yes |
| AutoGraph ("Flatten Graphs as Sequences") | **NeurIPS 2025** camera-ready | 2502.02216 | yes | yes |
| DMol | arXiv v3 Nov 2025 | 2504.06312 | yes (own re-runs) | yes (own re-runs) |
| **GraphBSI** | **ICLR 2026** (OpenReview `py2NPbAdvH`) | **2511.03015** (use **v2**) | yes | yes |
| GVT (Graph VQ-Transformer) | arXiv Dec 2025, AAAI format, unrefereed | 2512.02667 | yes — **Test split** | yes |
| LG-Flow ("Principled Latent Diffusion for Graphs via Laplacian Autoencoders") | arXiv Jan 2026 | 2601.13780 | yes | yes |
| MolHIT | arXiv Feb 2026 | 2602.17602 | yes (own re-runs) | yes — **unfiltered** |
| Flowette ("Flow Matching with Graphette Priors") | arXiv Feb 2026 | 2602.23566 | partial | yes |

### 2a. MOSES — DiGress-lineage protocol (FCD/SNN/Scaf vs **TestSF**)

Steps = sampling/denoising steps. "—" = not reported.

| Method | Class | Steps | Val.↑ | Uniq.↑ | Nov.↑ | Filters↑ | FCD↓ | SNN↑ | Scaf↑ | Params | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Training set | — | — | 100.0 | 100.0 | 0.0 | 100.0 | 0.01* | 0.64* | 99.1* | — | DeFoG/GraphBSI tables (*Test-split values — inconsistent with the model rows; MolHIT/FragFM correctly print 0.48 / 0.59 / 0.0) |
| VAE | SMILES | — | 97.7 | 99.8 | 69.5 | 99.7 | 0.57 | 0.58 | 5.9 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| CharRNN | SMILES | — | 97.5 | 99.9 | 84.2 | 99.4 | 0.52 | 0.56 | 11.0 | — | [2602.17602](https://arxiv.org/abs/2602.17602) (MolHIT) |
| JT-VAE | Fragment | — | 100.0 | 100.0 | 99.9 | 97.8 | 1.00 | 0.53 | 10.0 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| GraphINVENT | Autoreg. | — | 96.4 | 99.8 | — | 95.0 | 1.22 | 0.54 | 12.7 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| SAFE-GPT | Fragment seq. | — | 98.1 | 100.0 | 90.9 | 98.2 | 0.71 | 0.54 | 9.8 | — | [2502.15805](https://arxiv.org/abs/2502.15805) (FragFM) |
| SAFE-GPT | Fragment seq. | — | 99.8 | 98.9 | 43.7 | 97.7 | 0.72 | 0.57 | 6.3 | — | [2602.17602](https://arxiv.org/abs/2602.17602) (MolHIT) |
| GenMol | Fragment seq. | — | 99.7 | 64.0 | 68.9 | 98.1 | 16.4 | 0.64 | 1.6 | — | [2602.17602](https://arxiv.org/abs/2602.17602) |
| MolHF | One-shot flow | — | 71.0 | 100.0 | 100.0 | 17.8 | 35.4 | 0.23 | 1.5 | — | [2502.15805](https://arxiv.org/abs/2502.15805) |
| MolGrow | One-shot (frag.) | — | 100.0 | 100.0 | 99.8 | 82.7 | 6.39 | 0.42 | 7.7 | — | [2502.15805](https://arxiv.org/abs/2502.15805) |
| ConGress | One-shot diff. | 500 | 83.4 | 99.9 | 96.4 | 94.8 | 1.48 | 0.50 | 16.4 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| **DiGress** | One-shot diff. | 500 | 85.7 | 100.0 | 95.0 | 97.1 | 1.19 | 0.52 | 14.8 | — | [2209.14734](https://arxiv.org/abs/2209.14734) ICLR'23 |
| DiGress (MolHIT re-run) | One-shot diff. | 500 | 87.1 | 100.0 | 94.2 | 97.5 | 1.25 | 0.53 | 12.8 | — | [2602.17602](https://arxiv.org/abs/2602.17602) |
| DiGress (CADD) | One-shot diff. | 500 | 92.2 | 82.3 | 74.2 | 76.2 | 37.19 | 0.24 | 0.0 | — | [2511.03015v2](https://arxiv.org/abs/2511.03015) ext. table |
| DisCo-MPNN | One-shot diff. (CTMC) | from {50,100,500} | 83.9 | 100.0 | 98.8 | 87.3 | 1.63 | 0.48 | 13.5 | 3/5/8 layers, hidden 256/512 | [2405.11416](https://arxiv.org/abs/2405.11416) |
| **DisCo**-GT | One-shot diff. (CTMC) | from {50,100,500} (DMol says 500) | 88.3 | 100.0 | 97.7 | 95.6 | 1.44 | 0.50 | 15.1 | DiGress GT verbatim | [2405.11416](https://arxiv.org/abs/2405.11416) |
| **Cometh** | One-shot diff. (CTMC) | 500 | 87.0±0.2 | — | — | 97.2±0.1 | 1.44±0.02 | 0.51±0.0 | 15.9±0.8 | — | [2406.06449](https://arxiv.org/abs/2406.06449) (Val&Uni 86.9, VUN 83.8) |
| **Cometh-PC** | One-shot diff. (CTMC) | 500 | 90.5±0.3 | 99.9 | (92.6) | 99.1±0.1 | 1.27±0.02 | 0.54±0.0 | 16.0±0.7 | — | [2406.06449](https://arxiv.org/abs/2406.06449) (Val&Uni 90.4, VUN 83.7) |
| Cometh (FragFM re-run) | One-shot diff. | 50 | 82.9 | — | — | 94.6 | 1.54 | 0.49 | 18.4 | — | [2502.15805](https://arxiv.org/abs/2502.15805) |
| Cometh (FragFM re-run) | One-shot diff. | 10 | 26.1 | — | — | 59.9 | 7.88 | 0.36 | 8.9 | — | [2502.15805](https://arxiv.org/abs/2502.15805) |
| **DeFoG** | One-shot flow matching | 50 | 83.9 | 99.9 | 96.9 | 96.5 | 1.87 | 0.50 | **23.5** | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| **DeFoG** | One-shot flow matching | 500 | 92.8 | 99.9 | 92.1 | 98.9 | 1.95 | 0.55 | 14.4 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| **GraphBFN** | One-shot BFN | 500 | **88.5** (GraphBSI misprints 98.5) | 99.8 | 89.0 | 98.3 | 1.07 | **0.59** | 10.0 | — | PMLR v267:56363 (ICML'25), Table 3 |
| **Pard** | AR + blockwise diffusion | not stated for MOSES (QM9 ablation: 140 total, 20/block) | 86.8 | 100 | 78.2 | 99.0 | 1.00 | 0.56 | 2.2 | — | [2402.03687](https://arxiv.org/abs/2402.03687) NeurIPS'24 |
| **AutoGraph** | AR on flattened graph seq. | AR (top-k 5, τ=1.0) | 87.4 | 100 | 85.9 | 98.6 | 0.91 | 0.55 | 10.2 | **≈113M** | [2502.02216](https://arxiv.org/abs/2502.02216) NeurIPS'25 |
| **DMol** | Discrete diffusion | **avg. 38** (adaptive, ANS) | 87.8 | 100 | **100** | 97.8 | 1.12 | 0.58 | 14.8 | — | [2504.06312](https://arxiv.org/abs/2504.06312) |
| **LGD** | Latent diffusion | — | 97.4 | 100 | 95.9 | — | 1.42 | — | — | — | [2402.02518](https://arxiv.org/abs/2402.02518) (no Filters/SNN/Scaf) |
| **LG-Flow** | Latent flow (Laplacian AE) | — | 88.4 | 99.9 | 90.5 | 98.9 | 1.43 | 0.55 | 12.3 | — | [2601.13780](https://arxiv.org/abs/2601.13780) |
| **GEEL** | AR edge-list | — | 92.1 | 100 | 81.1 | 97.5 | 1.28 | 0.52 | 3.6 | — | via [2501.01073](https://arxiv.org/abs/2501.01073) (G2PT) |
| **G2PT**-small | AR seq. on graph tokens | — | 95.1 | 100 | 91.7 | 97.4 | 1.10 | 0.52 | 5.0 | 10M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| **G2PT**-base | AR seq. | — | 96.4 | 100 | 86.0 | 98.3 | **0.97** | 0.55 | 3.3 | 85M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| **G2PT**-large | AR seq. | — | 97.2 | 100 | 79.4 | 98.9 | 1.02 | 0.55 | 2.9 | 300M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| LigGPT | SMILES AR | — | 90.0 | 99.9 | 94.1 | — | — | — | — | — | via [2501.01073](https://arxiv.org/abs/2501.01073) |
| **FragFM** | Fragment flow matching | 50 | 99.5 | — | — | 98.5 | 0.65 | 0.54 | 11.2 | ~21.7M† | [2502.15805](https://arxiv.org/abs/2502.15805) |
| **FragFM** | Fragment flow matching | 300 | 99.8 | 100.0 | 87.1 | 99.1 | **0.58** | 0.56 | 10.9 | ~21.7M† | [2502.15805](https://arxiv.org/abs/2502.15805) (main result) |
| **MolHIT** | Hierarchical disc. diff. | — | 99.1 | 99.8 | 91.4 | 98.0 | 1.03 | 0.55 | 14.4 | — | [2602.17602](https://arxiv.org/abs/2602.17602) |
| **Flowette** | Flow matching, graphette prior | — | — | 99.9 | 98.10 | — | — | 0.58 | 15.3 | — | [2602.23566](https://arxiv.org/abs/2602.23566) (no MOSES Validity / Filters / FCD) |
| **GraphBSI (EM)** | One-shot BSI | 10 | 90.9 | 100.0 | 99.2 | 85.4 | 3.74 | 0.43 | 13.7 | — | [2511.03015v2](https://arxiv.org/abs/2511.03015) ext. |
| **GraphBSI (OU)** | One-shot BSI | 10 | 94.4 | 100.0 | 98.9 | 89.0 | 3.88 | 0.45 | 14.5 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 20 | 97.5 | 100.0 | 97.9 | 93.6 | 1.83 | 0.47 | 15.7 | — | v2 ext. |
| **GraphBSI (OU)** | One-shot BSI | 20 | 98.2 | 100.0 | 97.8 | 94.5 | 1.92 | 0.48 | 14.4 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 50 | 99.3 | 100.0 | 96.5 | 96.9 | 1.06 | 0.50 | 15.2 | — | v2 main |
| **GraphBSI (OU)** | One-shot BSI | 50 | 99.7 | 100.0 | 94.6 | 98.2 | 1.19 | 0.52 | 15.1 | — | v2 main |
| GraphBSI (γ→∞) | One-shot BSI | 50 | 99.9 | 99.9 | 89.9 | 99.2 | 1.58 | 0.56 | 11.7 | — | v2 ext. |
| GraphBSI (γ→∞, FP) | One-shot BSI | 50 | 99.9 | 99.9 | 89.7 | 99.1 | 1.06 | 0.56 | 13.1 | — | v2 ext. |
| GraphBSI (FB) | One-shot BSI | 50 | 99.6 | 100.0 | 95.9 | 97.5 | 1.15 | 0.51 | 15.0 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 500 | 99.8 | 100.0 | 92.5 | 99.1 | **0.72** | 0.54 | 14.3 | — | v2 main |
| **GraphBSI (OU)** | One-shot BSI | 500 | 99.9 | 100.0 | 90.7 | 99.2 | 0.90 | 0.55 | 12.7 | — | v2 main |
| GraphBSI (EM, lin) | One-shot BSI | 500 | 99.8 | 100.0 | 91.9 | 99.1 | 0.85 | 0.54 | 12.3 | — | v2 ext. |
| GraphBSI (OU, lin) | One-shot BSI | 500 | 99.9 | 99.9 | 90.6 | 99.2 | 0.98 | 0.55 | 14.2 | — | v2 ext. |
| GraphBSI (FB, β₀=1) | One-shot BSI | 500 | 100.0 | 99.6 | 80.7 | 99.6 | 2.84 | 0.59 | 8.7 | — | v2 ext. |

† FragFM reports 4,257,290 params (fragment-to-vector encoder) + 17,418,274 params (coarse graph
propagation module, Graph Transformer backbone, 5 layers, 8 heads, node dim 256, edge dim 128).

### 2b. MOSES — not on the TestSF protocol (do **not** mix with 2a)

| Method | Val.↑ | Uniq.↑ | Nov.↑ | Filters↑ | FCD↓ | SNN↑ | Scaf↑ | Note |
|---|---|---|---|---|---|---|---|---|
| **GVT** (Graph VQ-Transformer) | 99.42 | 100 | 61.43 | 99.59 | **0.16** | **0.64** | 3.4 | [2512.02667](https://arxiv.org/abs/2512.02667). FCD 0.16 is **below the TestSF floor (0.4755)** and SNN 0.64 equals MOSES `Train` SNN/**Test** (0.6419) → these are Test-split numbers, not TestSF. Also the paper mislabels the metrics ("SNN = Scaffold Similarity", "Scaf = Scaffold Uniqueness"). 3-author preprint, Dec 2025, unreviewed. **Treat as not comparable.** |
| **GraphBSI v1 (EM, 500)** | 99.8 | 100.0 | 92.3 | 99.1 | 0.26 | 0.57 | 86.8 | Superseded by v2. Test-split; see Trap 2. |
| **GraphBSI v1 (OU, 500)** | 99.9 | 100.0 | 90.3 | 99.3 | 0.41 | 0.59 | 88.2 | Superseded by v2. |
| MOSES official leaderboard | see §1b | | | | | | | Reports both Test and TestSF columns |

**GEEL (ICLR 2024, [2312.02230](https://arxiv.org/abs/2312.02230), Table 14) uses a third protocol**:
columns are Validity / NSPDK / FCD / Scaf / SNN / Frag, with Scaf a **0–1 cosine similarity against the
Test set** and validity computed *after* SMILES conversion with different sanitisation:

| Method | Validity | NSPDK↓ | FCD↓ | Scaf↑ (0–1, Test) | SNN↑ | Frag↑ |
|---|---|---|---|---|---|---|
| DiGress | 100.00 | 0.0005 | 0.6788 | 0.8653 | 0.5488 | 0.9988 |
| GEEL | 99.99 | 0.0001 | 0.1603 | 0.8622 | 0.6310 | 0.9996 |

Note DiGress's validity here is **100.00**, versus 85.7 in its own paper — entirely a sanitisation
artifact. **Do not compare any of these numbers to §2a.**

Full MOSES official leaderboard (mean±std over 3 runs, from
[github.com/molecularsets/moses](https://github.com/molecularsets/moses) README; paper
[1811.12823](https://arxiv.org/abs/1811.12823), Frontiers in Pharmacology 2020):

| Model | Valid | Unique@1k | Unique@10k | FCD/Test | FCD/TestSF | SNN/Test | SNN/TestSF | Frag/Test | Frag/TestSF | Scaf/Test | Scaf/TestSF | IntDiv | IntDiv2 | Filters | Novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Train* | 1.0 | 1.0 | 1.0 | 0.008 | 0.4755 | 0.6419 | 0.5859 | 1.0 | 0.9986 | 0.9907 | 0.0 | 0.8567 | 0.8508 | 1.0 | 1.0‡ |
| HMM | 0.076±0.0322 | 0.623±0.1224 | 0.5671±0.1424 | 24.4661±2.5251 | 25.4312±2.5599 | 0.3876±0.0107 | 0.3795±0.0107 | 0.5754±0.1224 | 0.5681±0.1218 | 0.2065±0.0481 | 0.049±0.018 | 0.8466±0.0403 | 0.8104±0.0507 | 0.9024±0.0489 | 0.9994±0.001 |
| NGram | 0.2376±0.0025 | 0.974±0.0108 | 0.9217±0.0019 | 5.5069±0.1027 | 6.2306±0.0966 | 0.5209±0.001 | 0.4997±0.0005 | 0.9846±0.0012 | 0.9815±0.0012 | 0.5302±0.0163 | 0.0977±0.0142 | 0.8738±0.0002 | 0.8644±0.0002 | 0.9582±0.001 | 0.9694±0.001 |
| Combinatorial | 1.0±0.0 | 0.9983±0.0015 | 0.9909±0.0009 | 4.2375±0.037 | 4.5113±0.0274 | 0.4514±0.0003 | 0.4388±0.0002 | 0.9912±0.0004 | 0.9904±0.0003 | 0.4445±0.0056 | 0.0865±0.0027 | 0.8732±0.0002 | 0.8666±0.0002 | 0.9557±0.0018 | 0.9878±0.0008 |
| CharRNN | 0.9748±0.0264 | 1.0±0.0 | 0.9994±0.0003 | **0.0732±0.0247** | 0.5204±0.0379 | 0.6015±0.0206 | 0.5649±0.0142 | 0.9998±0.0002 | 0.9983±0.0003 | 0.9242±0.0058 | **0.1101±0.0081** | 0.8562±0.0005 | 0.8503±0.0005 | 0.9943±0.0034 | 0.8419±0.0509 |
| AAE | 0.9368±0.0341 | 1.0±0.0 | 0.9973±0.002 | 0.5555±0.2033 | 1.0572±0.2375 | 0.6081±0.0043 | 0.5677±0.0045 | 0.991±0.0051 | 0.9905±0.0039 | 0.9022±0.0375 | 0.0789±0.009 | 0.8557±0.0031 | 0.8499±0.003 | 0.996±0.0006 | 0.7931±0.0285 |
| VAE | 0.9767±0.0012 | 1.0±0.0 | 0.9984±0.0005 | 0.099±0.0125 | 0.567±0.0338 | **0.6257±0.0005** | **0.5783±0.0008** | 0.9994±0.0001 | 0.9984±0.0003 | **0.9386±0.0021** | 0.0588±0.0095 | 0.8558±0.0004 | 0.8498±0.0004 | **0.997±0.0002** | 0.6949±0.0069 |
| JTN-VAE | 1.0±0.0 | 1.0±0.0 | 0.9996±0.0003 | 0.3954±0.0234 | 0.9382±0.0531 | 0.5477±0.0076 | 0.5194±0.007 | 0.9965±0.0003 | 0.9947±0.0002 | 0.8964±0.0039 | 0.1009±0.0105 | 0.8551±0.0034 | 0.8493±0.0035 | 0.976±0.0016 | 0.9143±0.0058 |
| LatentGAN | 0.8966±0.0029 | 1.0±0.0 | 0.9968±0.0002 | 0.2968±0.0087 | 0.8281±0.0117 | 0.5371±0.0004 | 0.5132±0.0002 | 0.9986±0.0004 | 0.9972±0.0007 | 0.8867±0.0009 | 0.1072±0.0098 | 0.8565±0.0007 | 0.8505±0.0006 | 0.9735±0.0006 | 0.9498±0.0006 |

‡ The README's `Train` Novelty = 1.0 is an error; the paper prints 0.0. LatentGAN differs slightly
between the README and the paper (paper: FCD/Test 0.296±0.021, Scaf/TestSF 0.1±0.006, Unique@10k
0.997±0.005); the README looks more precise/updated. The paper rounds CharRNN/VAE/AAE/LatentGAN to 3
decimals.

### 2c. GuacaMol distribution-learning

Original paper scale is [0,1]; graph papers use 0–100. Both shown as printed.

| Method | Class | Steps | Validity↑ | Uniq. / V.U.↑ | Novelty / V.U.N.↑ | KL div↑ | FCD score↑ | Params | Source |
|---|---|---|---|---|---|---|---|---|---|
| Training set | — | — | 100.0 | 100.0 | 0.0 | 99.9 | 92.8 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| Random SMILES sampler | — | — | 1.000 | 0.997 | 0.000 | 0.998 | 0.929 | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| SMILES LSTM | SMILES | — | 0.959 | 1.000 | 0.912 | 0.991 | **0.913** | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| SMILES LSTM (as V.U./V.U.N.) | SMILES | — | 95.9 | 95.9 | 87.4 | 99.1 | 91.3 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| Graph MCTS | Search | — | 1.000 | 1.000 | 0.994 | 0.522 | 0.015 | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| AAE | SMILES | — | 0.822 | 1.000 | 0.998 | 0.886 | 0.529 | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| ORGAN | SMILES GAN | — | 0.379 | 0.841 | 0.687 | 0.267 | 0.000 | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| VAE | SMILES | — | 0.870 | 0.999 | 0.974 | 0.982 | 0.863 | — | [1811.09621](https://arxiv.org/abs/1811.09621) |
| NAGVAE | One-shot | — | 92.9 | 88.7 | 88.7 | 38.4 | 0.9 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| MCTS | Search | — | 100.0 | 100.0 | 95.4 | 82.2 | 1.5 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| ConGress | One-shot diff. | 500 | 0.1 | 100 | 100 | 36.1 | 0.0 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| **DiGress** | One-shot diff. | 500 | 85.2 | 85.2 | 85.1 | 92.9 | 68.0 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| DisCo-MPNN | One-shot diff. | 500 | 68.7 | 100.0 | 96.4 | 77.0 | 36.4 | — | [2405.11416](https://arxiv.org/abs/2405.11416) |
| **DisCo**-GT | One-shot diff. | 500 | 86.6 | 86.6 | 86.5 | 92.6 | 59.7 | — | [2405.11416](https://arxiv.org/abs/2405.11416) |
| **Cometh** | One-shot diff. | 500 | 94.4±0.2 | 94.4±0.2 | 93.5±0.3 | 94.1±0.4 | 67.4±0.3 | — | [2406.06449](https://arxiv.org/abs/2406.06449) |
| **Cometh-PC** | One-shot diff. | 500 | 98.9±0.1 | 98.9±0.1 | 97.6±0.2 | 96.7±0.2 | 72.7±0.2 | — | [2406.06449](https://arxiv.org/abs/2406.06449) |
| **DeFoG** | Flow matching | 50 | 91.7 | 91.7 | 91.2 | 92.3 | 57.9 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| **DeFoG** | Flow matching | 500 | 99.0 | 99.0 | 97.9 | 97.7 | 73.8 | — | [2410.04263](https://arxiv.org/abs/2410.04263) |
| **LG-Flow** | Latent flow | — | 93.0 | 93.0 | 91.5 | 97.2 | 81.8 | — | [2601.13780](https://arxiv.org/abs/2601.13780) |
| **FragFM** | Fragment flow | — | 99.7 | 99.3 | 95.0 | 97.4 | **85.8** | ~21.7M | [2502.15805](https://arxiv.org/abs/2502.15805) |
| **Flowette** | Flow matching | — | 98.6 | — | 98.0 | 97.9 | — | — | [2602.23566](https://arxiv.org/abs/2602.23566) |
| **MolHIT** | Hierarchical diff. | — | 87.1 | 87.1 | 86.0 | 96.7 | 54.9 | — | [2602.17602](https://arxiv.org/abs/2602.17602) — **unfiltered** GuacaMol, not comparable |
| DiGress (full/unfiltered) | One-shot diff. | 500 | 74.7 | 74.6 | 74.0 | 92.4 | 61.1 | — | [2602.17602](https://arxiv.org/abs/2602.17602) |
| **GEEL** | AR edge-list | — | 88.2 | 98.2 | 89.1 | 93.1 | 71.5 | — | via [2501.01073](https://arxiv.org/abs/2501.01073) |
| **AutoGraph** | AR graph seq. | AR | 91.6 | (Uniq 100) | (Novel 97.7) | 97.5 | 79.2 | ≈113M | [2502.02216](https://arxiv.org/abs/2502.02216) NeurIPS'25 |
| **AutoGraph\*** (PubChem-10M pretrain) | AR graph seq. | AR | 95.9 | (Uniq 100) | (Novel 95.5) | 98.1 | **91.4** (94.7 with newer `fcd` pkg) | ≈113M | [2502.02216](https://arxiv.org/abs/2502.02216) |
| **DMol** | Discrete diffusion | **avg. 48** (ANS) | 86.7 | (Uniq 100) | (Novel 100) | 94.2 | 69.3 | — | [2504.06312](https://arxiv.org/abs/2504.06312) |
| **G2PT**-small | AR seq. | — | 90.4 | 100 | 99.8 | 92.8 | 86.6 | 10M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| **G2PT**-base | AR seq. | — | 94.6 | 100 | 99.5 | 96.0 | **93.4** | 85M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| **G2PT**-large | AR seq. | — | 95.3 | 100 | 99.5 | 95.6 | 92.7 | 300M | [2501.01073](https://arxiv.org/abs/2501.01073) |
| LigGPT | SMILES AR | — | 98.6 | 99.8 | 100 | — | — | — | via [2501.01073](https://arxiv.org/abs/2501.01073) |
| **GVT** | VQVAE + AR | — | 97.46 | 99.9 | 82.39 | **99.61** | 85.50 | — | [2512.02667](https://arxiv.org/abs/2512.02667) — unreviewed preprint |
| MolGPT 25.5M | SMILES AR | — | 0.9807 | 0.9914 | 0.9900 | 0.9924 | 0.9085 | 25.5M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| MolGPT 85.4M | SMILES AR | — | 0.9816 | 0.9922 | 0.9896 | 0.9929 | 0.9096 | 85.4M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| MolGram 27.2M | SMILES AR + n-gram | — | 0.9821 | 0.9942 | 0.9892 | 0.9931 | 0.9101 | 27.2M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| MolGram 87.5M | SMILES AR + n-gram | — | 0.9829 | 0.9946 | 0.9894 | **0.9935** | 0.9112 | 87.5M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| FO-ARM | Insertion AR | — | 98.5 | 98.3 | 80.4 | **99.4** | 90.3 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| AO-ARM | Insertion AR | — | 83.1 | 83.1 | 80.7 | 93.1 | 69.3 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| LO-ARM | Insertion AR | — | 89.7 | 89.4 | 81.6 | 96.1 | 83.7 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| AO-IP | Insertion | — | 88.2 | 88.2 | 87.7 | 95.3 | 79.1 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| IP (policy term.) | Insertion | — | 97.2 | 97.0 | 94.9 | 97.1 | 88.2 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| IP (classifier term.) | Insertion | — | 97.4 | 97.3 | 95.6 | 97.2 | 89.2 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| VAE (re-reported) | SMILES | — | 87.0 | 86.9 | 84.7 | 98.2 | 86.3 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| AAE (re-reported) | SMILES | — | 82.2 | 82.2 | 82.1 | 88.6 | 52.9 | — | [2606.02133](https://arxiv.org/abs/2606.02133) *(unverified)* |
| EDM-SYCo | Lift-to-3D diff. | — | — | — | — | 0.93±0.006 | 0.79±0.002 | — | [2406.10513](https://arxiv.org/abs/2406.10513), [0,1] scale *(unverified)* |
| MoLeR | Fragment | — | — | — | — | 0.964 | 0.625 | — | via [2302.01129](https://arxiv.org/abs/2302.01129) *(unverified)* |
| MiCaM | Fragment | — | — | — | — | 0.989 | 0.731 | — | via [2302.01129](https://arxiv.org/abs/2302.01129) *(unverified)* |
| **GraphBSI (EM)** | One-shot BSI | 10 | 86.6 | 86.6 | 86.5 | 85.5 | 27.6 | — | [2511.03015v2](https://arxiv.org/abs/2511.03015) ext. |
| **GraphBSI (OU)** | One-shot BSI | 10 | 91.9 | 91.9 | 91.8 | 84.5 | 24.2 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 20 | 97.5 | 97.5 | 97.3 | 87.5 | 40.7 | — | v2 ext. |
| **GraphBSI (OU)** | One-shot BSI | 20 | 97.1 | 97.1 | 96.8 | 89.3 | 49.7 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 50 | 97.5 | 97.5 | 97.2 | 90.7 | 65.6 | — | v2 main |
| **GraphBSI (OU)** | One-shot BSI | 50 | 99.2 | 99.2 | 98.7 | 93.7 | 71.3 | — | v2 main |
| GraphBSI (γ→∞) | One-shot BSI | 50 | 99.6 | 99.6 | 98.3 | 95.1 | 61.4 | — | v2 ext. |
| GraphBSI (γ→∞, FP) | One-shot BSI | 50 | 99.6 | 99.6 | 98.3 | 97.4 | 75.1 | — | v2 ext. |
| **GraphBSI (EM)** | One-shot BSI | 500 | 98.8 | 98.8 | 98.3 | 94.6 | **82.6** | — | v2 main |
| **GraphBSI (OU)** | One-shot BSI | 500 | 99.6 | 99.6 | 98.2 | **98.4** | 80.3 | — | v2 main |

Checked and confirmed **not** to report GuacaMol distribution-learning: SAFE-GPT (2310.10773), Pard
(2402.03687), GenMol (2501.06158), GraphBFN (PMLR v267 / 2510.10211), LO-ARM (2503.05979), MolMIM,
Chemformer, STONED, GLAD, LGD, Graph DiT. The official
[guacamol_baselines](https://github.com/BenevolentAI/guacamol_baselines) repo has **no leaderboard** —
the 1811.09621 table is the only authoritative baseline source.

Checked by full-text grep and found to have **no MOSES and no GuacaMol numbers at all** (zero hits
outside the bibliography):

| Method | arXiv | Venue | Evaluates on instead |
|---|---|---|---|
| GruM (Graph Generation with Diffusion Mixture) | [2302.03596](https://arxiv.org/abs/2302.03596) | ICML 2024 | QM9, ZINC250k |
| SwinGNN | [2307.01646](https://arxiv.org/abs/2307.01646) | TMLR | synthetic, QM9/ZINC250k |
| CatFlow (Variational Flow Matching) | [2406.04843](https://arxiv.org/abs/2406.04843) | NeurIPS 2024 | QM9, ZINC250k |
| GraphARM | [2307.08849](https://arxiv.org/abs/2307.08849) | ICML 2023 | QM9, ZINC250k |
| HGGT (K²-trees) | [2305.19125](https://arxiv.org/abs/2305.19125) | ICLR 2024 | synthetic, QM9/ZINC250k |
| GLAD | [2403.16883](https://arxiv.org/abs/2403.16883) | — | QM9 etc. |
| Fisher-Flow | [2405.14664](https://arxiv.org/abs/2405.14664) | — | QM9 |
| EDGE | [2305.04111](https://arxiv.org/abs/2305.04111) | ICML 2023 | MOSES appears only as a citation |
| MolHF | [2305.08457](https://arxiv.org/abs/2305.08457) | IJCAI 2023 | its MOSES row in §2a comes only from FragFM's table |
| VBFN (Variational Bayesian Flow Network) | [2601.22524](https://arxiv.org/abs/2601.22524) | Jan 2026 | QM9/ZINC250k |
| BWFlow (Bures-Wasserstein Flow Matching) | [2506.14020](https://arxiv.org/abs/2506.14020) | ICLR 2026 | QM9/ZINC250k, 3D |
| Hierarchical BFN | [2510.10211](https://arxiv.org/abs/2510.10211) | withdrawn v2 | QM9, ZINC250k |
| MoFlow / GraphAF / GraphDF / GDSS / EDP-GNN / SPECTRE | — | — | appear only as QM9/ZINC250k/synthetic baselines |

So the complete set of methods that report MOSES **and/or** GuacaMol is: DiGress, ConGress, DisCo,
Cometh(-PC), DeFoG, GraphBFN, GraphBSI, Pard, LGD, LG-Flow, FragFM, Flowette, MolHIT, DMol, AutoGraph,
GEEL, G2PT, GVT, plus the SMILES/fragment baselines (VAE, CharRNN, AAE, JT-VAE/JTN-VAE, LatentGAN,
HMM, NGram, Combinatorial, GraphINVENT, MolGrow, SAFE-GPT, GenMol, LigGPT, MolGPT, MolGram, SMILES
LSTM, Graph MCTS, ORGAN, NAGVAE) and the insertion/AR variants in §2c.

### 2d. QM9

| Method | Steps | no-H Val.↑ | no-H Uniq.↑ | no-H FCD↓ | with-H Val.↑ | with-H Uniq.↑ | with-H FCD↓ | Source |
|---|---|---|---|---|---|---|---|---|
| Train Set | — | 99.3 | 100.0 | 0.05 | 99.3 | 100.0 | 0.05 | [2511.03015v2](https://arxiv.org/abs/2511.03015) |
| DiGress | 500 | 99.0 | 96.2 | — | 95.4±1.1 | 97.6±0.4 | — | [2209.14734](https://arxiv.org/abs/2209.14734) |
| DiGress (CADD) | 500 | 96.3 | 83.4 | 5.25 | — | — | — | v2 |
| DisCo | 500 | 99.3±0.6 | — | — | — | — | — | [2405.11416](https://arxiv.org/abs/2405.11416) |
| Fisher FM | 500 | 95.3 | — | — | — | — | — | v2 |
| Cometh | 500 | 99.6±0.1 | 96.8±0.2 | 0.25±0.01 | — | — | — | [2406.06449](https://arxiv.org/abs/2406.06449) |
| DeFoG | 50 | 98.9±0.1 | 96.2±0.2 | 0.26±0.00 | 97.1±0.0 | 98.1±0.0 | 0.31±0.00 | [2410.04263](https://arxiv.org/abs/2410.04263) |
| DeFoG | 500 | 99.3±0.0 | 96.3±0.3 | 0.12±0.00 | 98.0±0.0 | 96.7±0.0 | **0.05±0.00** | [2410.04263](https://arxiv.org/abs/2410.04263) |
| G2PT-small | — | 99.0 | 96.7 | **0.06** | — | — | — | [2501.01073](https://arxiv.org/abs/2501.01073) |
| G2PT-base | — | 99.0 | **96.8** | **0.06** | — | — | — | [2501.01073](https://arxiv.org/abs/2501.01073) |
| G2PT-large | — | 98.9 | 96.7 | **0.06** | — | — | — | [2501.01073](https://arxiv.org/abs/2501.01073) |
| **GraphBSI** | 50 | **99.9** | 93.7 | 0.30 | — | — | — | v2 |
| **GraphBSI** | 500 | **99.9** | 96.2 | 0.09 | **99.8** | 96.6 | 0.08 | v2 |

G2PT's own table also quotes DisCo QM9 FCD 0.25 and Cometh 0.11, which differ from Cometh's own
0.25±0.01 — another cross-paper inconsistency.

### 2e. Synthetic graphs (context for GraphBSI's weak SBM result)

| Method | Steps | Planar V.U.↑ | Planar Ratio↓ | Tree V.U.↑ | Tree Ratio↓ | SBM V.U.↑ | SBM Ratio↓ |
|---|---|---|---|---|---|---|---|
| Train set | — | 100.0 | 1.0 | 100.0 | 1.0 | 85.9 | 1.0 |
| HSpectre | 256 | 95.0 | 2.1 | 100.0 | 4.0 | 75.0 | 10.5 |
| DiGress | 1000 | 77.5 | 5.1 | 90.0 | 1.6 | 60.0 | 1.7 |
| DeFoG | 50 | 95.0±3.2 | 3.2±1.1 | 73.5±9.0 | 2.5±1.0 | 86.5±5.3 | 2.2±0.3 |
| DeFoG | 1000 | 99.5±1.0 | 1.6±1.0 | 96.5±2.6 | 1.6±0.4 | 90.0±5.1 | 4.9±1.3 |
| GraphBSI (EM) | 1000 | 100.0±0.0 | 3.8±1.0 | 96.5±3.7 | 1.3±0.4 | 50.5±4.6 | 11.3±1.4 |
| GraphBSI (OU) | 1000 | 100.0±0.0 | 3.2±0.6 | 100.0±0.0 | 1.8±0.5 | 77.5±2.7 | 4.6±1.1 |

Source: [2511.03015v2](https://arxiv.org/abs/2511.03015); 40 graphs × 5 runs (DeFoG protocol).
GraphBSI's SBM V.U. (77.5) is well below DeFoG (90.0) and even the train set (85.9) — clear headroom.

---

## 3. Sampling steps and parameter counts

### Sampling steps used by the leading graph models

| Model | Molecular (MOSES/GuacaMol) | Synthetic (planar/tree/SBM) | QM9 |
|---|---|---|---|
| DiGress | **500** — note the paper itself never states `T`; 500 is what DeFoG, GraphBSI, GraphBFN and DMol all attribute to it | **1000** | 500 |
| ConGress | 500 | 1000 | — |
| DisCo | not stated; "we select the number of sampling steps from {50, 100, 500}". DMol reports running it at **500** | — | 500 |
| Cometh | **500** (τ = 0.002). Full MOSES step ablation at 10/50/100/300/500/700/1000 | — | 500 |
| Cometh-PC | **500** τ-leaping steps **+ 10 corrector steps**, τ_c = 1.5 | — | 500 |
| DeFoG | **500** (full) and **50** (10% budget) | **1000** (full), 50 (reduced) | 500 / 50 |
| GraphBFN | not stated for MOSES (GraphBSI attributes 500); QM9/Planar ablation at 100/20/10/5% of 500 and 1000 | 1000 (Planar) | 500 |
| Pard | not stated for MOSES; QM9 ablation highlights **140 total** steps (20 per block) | — | 140 |
| FragFM | **300** for the headline MOSES result; also 10/50/100/500 reported | — | — |
| DMol | **adaptive**: average number of steps = **38** on MOSES, **48** on GuacaMol | — | — |
| MolHIT | not stated | — | — |
| **GraphBSI** | **50** and **500** in the main table; **10** and **20** in the extended table (grid over NFEs {25,50,100,200,500}) | **1000** (OU) | 50 / 500 |
| LG-Flow | not step-based in the same sense; 36.4× faster than DeFoG on GuacaMol, 58× on MOSES | — | — |
| Flowette | not stated | — | — |
| G2PT / GVT / GEEL / AutoGraph | autoregressive — token-by-token, no denoising steps | — | — |

GraphBSI's own step/noise hyperparameters (Table 5, v2): β_start 3.0, β_end 12.0 (GuacaMol/MOSES),
β(0) 1.0; γ_OU = 20.0 (GuacaMol) / 10.0 (MOSES) at 10% steps, and 250.0 (GuacaMol) / 90.0 (MOSES) at
100% steps; γ_Euler = 10.0 / 20.0 and 200.0 / 120.0 respectively. SBM/Planar/Tree use β_end 20.0.

Euler–Maruyama stability limit (GraphBSI Table 3): max stable γ = 12.94 (25 steps), 24.88 (50),
48.75 (100), 96.51 (200), 239.77 (500). This is why the EM scheme cannot use the high noise levels
that help SNN/Scaf — a concrete lever.

### Parameter counts

Parameter counts are **rarely reported** in this literature. What exists:

| Model | Params | Source |
|---|---|---|
| G2PT-small | ~10M (6 layers, 6 heads, d_model 384) | [2501.01073](https://arxiv.org/abs/2501.01073) |
| G2PT-base | ~85M (12 layers, 12 heads, d_model 768) | [2501.01073](https://arxiv.org/abs/2501.01073) |
| G2PT-large | ~300M (24 layers, 16 heads, d_model 1024) | [2501.01073](https://arxiv.org/abs/2501.01073) |
| G2PT (scaling study) | also 1M, 707M, 1.5B | [2501.01073](https://arxiv.org/abs/2501.01073) — MOSES/GuacaMol need >85M to be satisfying |
| FragFM | 4,257,290 (frag→vec encoder, 5-layer sparse MPNN) + 17,418,274 (coarse graph propagation, 5-layer Graph Transformer, 8 heads) | [2502.15805](https://arxiv.org/abs/2502.15805) |
| MolGPT | 25.5M / 85.4M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| MolGram | 27.2M / 87.5M | [2606.12113](https://arxiv.org/abs/2606.12113) *(unverified)* |
| AutoGraph | **≈113M** (LLaMA-S: 12 layers, 768 dims, 12 heads = GPT-2-small config). Scaling table: LLaMA-XS 25.2M, LLaMA-S 113M, LLaMA-M 402M | [2502.02216](https://arxiv.org/abs/2502.02216) |
| Flowette | L=4 attention layers (GuacaMol/MOSES), L=5 (QM9/ZINC250k); param count not given | [2602.23566](https://arxiv.org/abs/2602.23566) |
| DisCo-MPNN | 3/5/8 layers, hidden 256/512; total not given. DisCo-GT reuses DiGress's graph transformer verbatim | [2405.11416](https://arxiv.org/abs/2405.11416) |
| MolHIT | "we utilize the original graph transformer from DiGress, **maintaining the same model size**" | [2602.17602](https://arxiv.org/abs/2602.17602) |
| **DiGress / DeFoG / Cometh / DisCo / GraphBFN / GraphBSI / Pard / LGD / DMol / GVT / GEEL** | **not reported** | — |

DiGress, DeFoG and GraphBSI all share the same graph-transformer backbone family (GraphBSI: "the same
graph transformer architecture as [DeFoG, DiGress]"), so they are roughly parameter-matched; the
absence of counts means a parameter-efficiency claim would need your own measurement.

---

## 4. Concrete targets to beat

### MOSES (TestSF protocol — the one GraphBSI is in)

| Metric | GraphBSI v2 best | Target to beat (graph one-shot) | Target to beat (any model) | Headroom |
|---|---|---|---|---|
| Validity ↑ | 99.9 / 100.0 (FB) | — already SOTA | 100.0 (JT-VAE, hard-coded) | saturated |
| Uniqueness ↑ | 100.0 | — already SOTA | 100.0 | saturated |
| Novelty ↑ | 96.5 @50, 92.5 @500 | **100.0** (DMol, at Val. 87.8) / 98.1 (Flowette) | 100.0 | **real gap** |
| Filters ↑ | 99.2 @500 | 99.1 (Cometh-PC, FragFM) | 99.7 (VAE) | small |
| FCD ↓ | **0.72** | **0.58** (FragFM, fragments) / 0.91 (AutoGraph, AR) | **0.52** (CharRNN) | **real gap**; 0.72 is already best *atom-level one-shot graph* |
| SNN ↑ | 0.55 (0.59 for FB,β₀=1) | 0.59 (GraphBFN) | 0.59 | saturated at train value 0.5859 |
| Scaf ↑ | 15.7 @20 steps | **23.5** (DeFoG @50) | 23.5 | **biggest gap** |

### GuacaMol

| Metric | GraphBSI best | Target (graph one-shot) | Target (any) | Headroom |
|---|---|---|---|---|
| Validity ↑ | 99.6 | 99.7 (FragFM) | 100.0 (MCTS) | tiny |
| V.U. ↑ | 99.6 | 99.3 (FragFM) — already SOTA | 100.0 (G2PT) | tiny |
| V.U.N. ↑ | 98.7 | 98.0 (Flowette) — already SOTA | 98.7 | saturated |
| KL ↑ | 98.4 | 97.9 (Flowette) — already SOTA | **99.61** (GVT) / 99.4 (FO-ARM) | moderate |
| FCD score ↑ | **82.6** | **85.8** (FragFM, fragments) / 91.4 (AutoGraph\*, AR) | **93.4** (G2PT-base) | **biggest gap** |

### Priority ranking of the gaps

1. **MOSES Scaf/TestSF** — GraphBSI 15.7 vs DeFoG 23.5. The largest relative deficit, and *not*
   a saturation artifact (training set scores 0.0 here). GraphBSI's own ablation says Scaf benefits
   from **higher noise γ**, but EM becomes unstable above γ·Δt limits — so the OU scheme at high γ with
   few steps is the lever. Note the tension: DeFoG's 23.5 comes at Validity 83.9.
2. **GuacaMol FCD score** — 82.6 vs 85.8 (FragFM) / 91.4 (AutoGraph) / 93.4 (G2PT). GraphBSI's ablation
   says FCD is optimal at *medium* γ (20–100) and improves with ρ<1 (finer discretisation late).
   Caveat: the `fcd` package version shifts this score by ~3 points (AutoGraph: 91.4 → 94.7), so
   re-measure baselines with your own package version before claiming a record.
3. **MOSES FCD** — 0.72 vs 0.58 (FragFM, fragments) and the 0.52 CharRNN wall. GraphBSI already holds
   the **atom-level one-shot** record, so the framing "best atom-level graph FCD" is already defensible;
   beating 0.58 would make it unqualified among graph models.
4. **MOSES Novelty** — 92.5 @500 vs DMol 100.0 (at Val. 87.8) and Flowette 98.1. GraphBSI's ablation
   shows novelty *degrades* with more steps and more noise, so this trades directly against 1–3.
   DMol's 100.0 at Validity 87.8 suggests the frontier here is a validity/novelty trade-off curve, not
   a single point — consider reporting the curve.
5. **Synthetic SBM** — V.U. 77.5 vs DeFoG 90.0, Ratio 4.6 vs 2.2. Clear weakness.
6. **MOSES SNN and Validity, GuacaMol V.U./V.U.N.** — saturated; do not spend effort here.

### Claims to be careful about when writing

* Do **not** compare against GVT's MOSES FCD 0.16 / SNN 0.64 — those are Test-split numbers presented
  alongside TestSF baselines (same error GraphBSI v1 made), and the paper mislabels its metrics.
* Do **not** compare against MolHIT's GuacaMol numbers — unfiltered dataset.
* Do **not** repeat "GraphBFN validity 98.5" — the GraphBFN paper says 88.5.
* FragFM and MolGrow/JT-VAE/MiCaM/MoLeR are **fragment-level**; a fair "atom-level one-shot graph
  generation" claim can exclude them, but say so explicitly.
* G2PT/GVT/GEEL/LigGPT/MolGPT are **autoregressive**; GraphBSI's "one-shot" framing excludes them,
  but reviewers will ask about G2PT's MOSES FCD 0.97 and GuacaMol FCD 93.4.
* Flowette omits MOSES Validity, Filters and FCD entirely — its "98.1 novelty" is not accompanied by
  a validity number, so it may be a low-validity model.
* **GEEL's MOSES numbers are on a third protocol** (Scaf as 0–1 cosine vs Test, validity post-SMILES
  sanitisation giving DiGress 100.00). Never place GEEL's row next to the §2a rows.
* **Cometh's circulated "Novelty = 92.6"** does not appear in the Cometh paper, which reports
  Val/Val&Uni/VUN. It originates in DeFoG's table and propagates through GraphBSI, MolHIT and Flowette.
* **DeFoG's appendix** notes that *in its sampling-optimization figures* "the Guacamol and MOSES
  datasets are evaluated with 2,000 samples, and validity and novelty are computed using local
  implementations for efficiency instead of the original benchmarks". That caveat is scoped to the
  hyperparameter-search figures, not to its headline tables — but it is worth knowing.
* **Nobody in this literature reports parameter counts** except AutoGraph, G2PT and FragFM. If you want
  a parameter-efficiency claim, you will have to count DiGress/DeFoG/Cometh yourself from their configs.
* **Sample counts differ**: MOSES is evaluated with 25,000 samples by MolHIT and FragFM, 10,000 by the
  GuacaMol/DiGress lineage; Cometh and DeFoG average over 5 sampling runs, FragFM and MolHIT over 3,
  GraphBSI reports single runs for molecules (no error bars). GraphBSI adding error bars would be a
  cheap credibility win, since several gaps above are ~1 point.
