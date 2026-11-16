"""Print scored runs: python research/table.py RUNS_DIR [RUNS_DIR...]"""

import json
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    for m in sorted(root.rglob("metrics.json")):
        r = json.loads(m.read_text())
        h = r["headline"]
        print(f"{m.parent.relative_to(root.parent)!s:70s} " + " ".join(f"{k[:5]}={v:7.3f}" for k, v in h.items()))
