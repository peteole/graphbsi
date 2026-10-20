"""Published GraphBSI numbers (Tabs. 1 and 2 of the paper).

Values are stored exactly as printed in the paper so that
``scripts/reproduce.py`` can render a side-by-side comparison.

Percentages are stored as percentages (0-100), FCD on Moses is the raw FCD,
SNN is the raw score and the synthetic-graph ``ratio`` is the mean ratio.
"""

from __future__ import annotations

# --- Table 1: molecule generation -------------------------------------------
# GuacaMol: Val., V.U., V.U.N., KL, FCD (all in %, higher is better)
# Moses:    Val., Uniq., Nov., Filters, FCD (lower better), SNN, Scaf.
TABLE1 = {
    ("guacamol", "em_50"): {
        "validity": 97.5,
        "valid_unique": 97.5,
        "valid_unique_novel": 97.2,
        "kl": 90.7,
        "fcd": 65.6,
    },
    ("guacamol", "ou_50"): {
        "validity": 99.2,
        "valid_unique": 99.2,
        "valid_unique_novel": 98.7,
        "kl": 93.7,
        "fcd": 71.3,
    },
    ("guacamol", "em_500"): {
        "validity": 98.8,
        "valid_unique": 98.8,
        "valid_unique_novel": 98.3,
        "kl": 94.6,
        "fcd": 82.6,
    },
    ("guacamol", "ou_500"): {
        "validity": 99.6,
        "valid_unique": 99.6,
        "valid_unique_novel": 98.2,
        "kl": 98.4,
        "fcd": 80.3,
    },
    ("moses", "em_50"): {
        "validity": 99.3,
        "unique": 100.0,
        "novel": 96.5,
        "filters": 96.9,
        "fcd": 1.06,
        "snn": 0.50,
        "scaffold": 15.2,
    },
    ("moses", "ou_50"): {
        "validity": 99.7,
        "unique": 100.0,
        "novel": 94.6,
        "filters": 98.2,
        "fcd": 1.19,
        "snn": 0.52,
        "scaffold": 15.1,
    },
    ("moses", "em_500"): {
        "validity": 99.8,
        "unique": 100.0,
        "novel": 92.5,
        "filters": 99.1,
        "fcd": 0.72,
        "snn": 0.54,
        "scaffold": 14.3,
    },
    ("moses", "ou_500"): {
        "validity": 99.9,
        "unique": 100.0,
        "novel": 90.7,
        "filters": 99.2,
        "fcd": 0.90,
        "snn": 0.55,
        "scaffold": 12.7,
    },
}

# --- Table 2: synthetic graphs ----------------------------------------------
# V.U. in % (validity accuracy of the benchmark, which here equals valid and
# unique), mean ratio (lower is better). Paper reports mean +- std over the five
# batches of 40 generated graphs.
TABLE2 = {
    ("planar", "em_50"): {"vu": (7.5, 1.0), "ratio": (47.5, 4.3)},
    ("planar", "ou_50"): {"vu": (38.5, 8.6), "ratio": (18.0, 3.2)},
    ("planar", "em_1000"): {"vu": (100.0, 0.0), "ratio": (3.8, 1.0)},
    ("planar", "ou_1000"): {"vu": (100.0, 0.0), "ratio": (3.2, 0.6)},
    ("tree", "em_50"): {"vu": (89.0, 7.0), "ratio": (2.1, 0.8)},
    ("tree", "ou_50"): {"vu": (96.0, 1.2), "ratio": (2.5, 0.9)},
    ("tree", "em_1000"): {"vu": (96.5, 3.7), "ratio": (1.3, 0.4)},
    ("tree", "ou_1000"): {"vu": (100.0, 0.0), "ratio": (1.8, 0.5)},
    ("sbm", "em_50"): {"vu": (61.5, 5.8), "ratio": (4.2, 0.3)},
    ("sbm", "ou_50"): {"vu": (53.0, 7.5), "ratio": (51.4, 4.0)},
    ("sbm", "em_1000"): {"vu": (50.5, 4.6), "ratio": (11.3, 1.4)},
    ("sbm", "ou_1000"): {"vu": (77.5, 2.7), "ratio": (4.6, 1.1)},
}
