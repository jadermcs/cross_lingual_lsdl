"""Print held-out eval_loss per step for several training runs, side by side.

  uv run python scripts/compare_runs.py checkpoints/ablation/stage2-*
"""

import json
import sys
from pathlib import Path


def load(run):
    state = json.loads((Path(run) / "trainer_state.json").read_text())
    evals = {e["step"]: e["eval_loss"] for e in state["log_history"] if "eval_loss" in e}
    runtime = next((e["train_runtime"] for e in state["log_history"] if "train_runtime" in e), None)
    return evals, runtime


def main(runs):
    data = {Path(r).name: load(r) for r in runs}
    steps = sorted({s for evals, _ in data.values() for s in evals})
    w = max(len(n) for n in data)
    print("step".rjust(6), *(n.rjust(w) for n in data))
    for s in steps:
        print(str(s).rjust(6), *(f"{data[n][0][s]:.4f}".rjust(w) if s in data[n][0] else "-".rjust(w) for n in data))
    print("best".rjust(6), *(f"{min(e.values()):.4f}".rjust(w) for e, _ in data.values()))
    print("hours".rjust(6), *((f"{t / 3600:.2f}" if t else "-").rjust(w) for _, t in data.values()))


if __name__ == "__main__":
    main(sys.argv[1:])
