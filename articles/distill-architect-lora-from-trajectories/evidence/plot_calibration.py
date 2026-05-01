"""Bar chart showing per-iter agreement: did each proposer pick the same
knob (and the same value) as the ground-truth iter? Plus a sidebar with
mean per-proposal latency.

Reads race_results.json + preds.jsonl, writes calibration.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVIDENCE = Path(__file__).resolve().parent
RES = json.load(open(EVIDENCE / "race_results.json"))
PREDS = [json.loads(l) for l in open(EVIDENCE / "preds.jsonl")]

ITERS = [p["iter"] for p in PREDS]
gt_knob = [p["ground_truth"]["knob"] for p in PREDS]
gt_val = [p["ground_truth"]["new_value"] for p in PREDS]

def picks(side: str):
    rows = []
    for p in PREDS:
        s = p[side]
        if s.get("proposal") and s.get("valid", {}).get("valid"):
            knob = s["proposal"].get("knob")
            val = s["proposal"].get("new_value", s["proposal"].get("value"))
            rows.append((knob, val))
        else:
            rows.append((None, None))
    return rows

dpicks = picks("distilled")
npicks = picks("nim_8b")


def match_score(pick, knob, val):
    if pick == (None, None):
        return 0
    if pick[0] != knob:
        return 0
    if pick[1] != val:
        return 1   # knob match only
    return 2       # exact match


d_scores = [match_score(p, k, v) for p, k, v in zip(dpicks, gt_knob, gt_val)]
n_scores = [match_score(p, k, v) for p, k, v in zip(npicks, gt_knob, gt_val)]

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [3, 1]})
fig.patch.set_facecolor("#0f1115")

# left: per-iter score bars
ax = axes[0]
x = list(range(len(PREDS)))
w = 0.38

INDIGO = "#7C7CFF"
GRAY = "#9CA3AF"

ax.bar([i - w/2 for i in x], n_scores, width=w, label="8B NIM", color=GRAY, alpha=0.85)
ax.bar([i + w/2 for i in x], d_scores, width=w, label="3B distilled", color=INDIGO, alpha=0.95)

ax.set_xticks(x)
ax.set_xticklabels([f"#{i}\n{k}={v}" for i, k, v in zip(ITERS, gt_knob, gt_val)],
                   fontsize=8, color="#D1D5DB")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["miss", "knob only", "exact"], color="#D1D5DB")
ax.set_ylim(0, 2.4)
ax.set_facecolor("#0f1115")
for s in ax.spines.values():
    s.set_color("#374151")
ax.tick_params(colors="#D1D5DB")
ax.grid(True, axis="y", alpha=0.18, color="#374151")
ax.set_title("Per-iter behavioral cloning vs ground-truth proposal",
             color="#E5E7EB", fontsize=11, loc="left")
ax.legend(facecolor="#0f1115", edgecolor="#374151", labelcolor="#D1D5DB", fontsize=9)

# right: latency bars
ax2 = axes[1]
d_lat = RES["distilled"].get("mean_latency_s") or 0
n_lat = (RES["nim_8b"] or {}).get("mean_latency_s") or 0
ax2.bar(["8B NIM"], [n_lat * 1000], color=GRAY, alpha=0.85, width=0.55)
ax2.bar(["3B distilled"], [d_lat * 1000], color=INDIGO, alpha=0.95, width=0.55)
ax2.set_facecolor("#0f1115")
for s in ax2.spines.values():
    s.set_color("#374151")
ax2.tick_params(colors="#D1D5DB")
ax2.grid(True, axis="y", alpha=0.18, color="#374151")
ax2.set_ylabel("ms / proposal", color="#D1D5DB")
ax2.set_title("Throughput", color="#E5E7EB", fontsize=11, loc="left")
for x_, v in zip([0, 1], [n_lat * 1000, d_lat * 1000]):
    ax2.text(x_, v * 0.5, f"{v:.0f}", ha="center", color="#0f1115", fontsize=10, fontweight="bold")

speedup = RES.get("throughput_speedup")
if speedup:
    ax2.text(0.5, max(d_lat, n_lat) * 1000 * 1.05,
             f"{speedup:.1f}× speedup",
             transform=ax2.transData, ha="center", color=INDIGO, fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(EVIDENCE / "calibration.png", dpi=150, facecolor="#0f1115")
print(f"wrote {EVIDENCE/'calibration.png'}")
