"""Merge sharded grade outputs into one experiment dir and recompute metrics.
python3 validation/merge_shards.py <out_dir> <shard_dir> [<shard_dir> ...]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase2.metrics import compute_metrics

out = Path(sys.argv[1])
records = []
for s in sys.argv[2:]:
    records += json.load(open(Path(s) / "evaluations.json"))
records.sort(key=lambda r: (r["pipeline"], r["item_id"], r["context_level"], r["round"]))
(out / "evaluations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
(out / "drift_metrics.json").write_text(json.dumps(compute_metrics(records), indent=2), encoding="utf-8")
print(f"merged {len(records)} records from {len(sys.argv) - 2} shards into {out}")
