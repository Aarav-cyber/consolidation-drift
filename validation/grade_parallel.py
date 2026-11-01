"""Grade an experiment dir in parallel shards, one Groq key per shard, then merge.
python3 validation/grade_parallel.py <experiment_dir>
Shards live in <experiment_dir>_grade/ (symlinks to the chains); logs in logs/.
"""
import os, shutil, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
src = Path(sys.argv[1]); base = Path(str(src) + "_grade")
if base.exists():
    shutil.rmtree(base)
keys = []
for line in open(".env"):
    if line.startswith("GROQ_API_KEY="):
        keys.append(line.strip().split("=", 1)[1])
    if line.startswith("GROQ_API_KEYS_EXTRA="):
        keys += [k for k in line.strip().split("=", 1)[1].split(",") if k]
n = len(keys)
files = sorted(f for f in src.glob("*/*.json") if f.name not in ("evaluations.json", "drift_metrics.json"))
shards = [base / f"shard{i}" for i in range(n)]
for i, f in enumerate(files):
    d = shards[i % n] / f.parent.name
    d.mkdir(parents=True, exist_ok=True)
    (d / f.name).symlink_to(f.resolve())
procs = []
for i, sd in enumerate(shards):
    env = dict(os.environ, GROQ_API_KEY=keys[i],
               GROQ_API_KEYS_EXTRA=",".join(k for j, k in enumerate(keys) if j != i))
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "src.phase2.grade", "--dir", str(sd), "--model", "openai/gpt-oss-120b", "--delay", "1.0"],
        stdout=open(f"logs/{src.name}_grade_shard{i}.log", "w"), stderr=subprocess.STDOUT, env=env))
rc = [p.wait() for p in procs]
print("shard exit codes:", rc, flush=True)
if all(r == 0 for r in rc):
    subprocess.run([sys.executable, "validation/merge_shards.py", str(src)] + [str(s) for s in shards], check=True)
else:
    sys.exit(1)
