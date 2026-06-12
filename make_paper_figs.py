#!/usr/bin/env python3
"""Generate the restyled paper figures (fig2, fig3, fig4) from cached wandb data.

Inputs (created by make_figures.py / the rq1 fetch):
    figures/cluttered/raw_history.pkl   4 variants x 3 seeds, 15k episodes
    figures/fourrooms/raw_history.pkl   4 variants x 3 seeds, 15k episodes
    figures/rq1_runs.pkl                shared vs separate-network IPPO (30k episodes)

Outputs (paper slots, referenced from latex/main.tex):
    latex/figures/fig2_reward.* fig2_runtime.*   RQ1 shared vs separate encoder
    latex/figures/fig3.*                         RQ2 IPPO vs MAPPO, both envs (2x2)
    latex/figures/fig4_norm.*                    RQ4 GRPO normalization ablation

Style (fonts, palette, EMA smoothing, SEM bands) is imported from make_figures.py
so every figure in the paper matches fig1.
"""
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import make_figures as mf  # rcParams applied on import

X = "episode/x_axis"
REW = "episode/collective_reward_mean"
ENT = "train/entropy"
SMOOTH = 0.6
ENVS = [("cluttered", "Cluttered"), ("fourrooms", "FourRooms")]


def load_groups(env):
    with open(f"figures/{env}/raw_history.pkl", "rb") as f:
        blob = pickle.load(f)
    groups = {}
    for e in blob["selected"]:
        lbl = mf.variant_label(e["cfg"])
        groups.setdefault(lbl, []).append(e)
    return groups


def draw(ax, groups, labels, key, ylab, legend=False, title=None):
    for lbl in labels:
        agg = mf.aggregate(groups.get(lbl, []), key, X, smooth=SMOOTH)
        if agg is None:
            continue
        grid, mean, sem, _ = agg
        ax.plot(grid, mean, color=mf.PALETTE[lbl], ls=mf.LINESTYLE[lbl], lw=1.6, label=lbl)
        ax.fill_between(grid, mean - sem, mean + sem, color=mf.PALETTE[lbl], alpha=0.18, lw=0)
    ax.set_xlabel("Episodes")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)
    if legend:
        ax.legend(frameon=False, handlelength=1.9)


def save(fig, stem, pad=0.3):
    fig.tight_layout(pad=pad)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ---------------- fig2: RQ1 shared vs separate encoder (Cluttered, 30k eps) ---
with open("figures/rq1_runs.pkl", "rb") as f:
    rq1 = pickle.load(f)
shared = next(r for r in rq1 if r["name"] == "IPPO(shared backbone)")
indp = next(r for r in rq1 if r["name"] == "IPPO (indp network)" and r["state"] == "finished")
RQ1 = [
    ("IPPO (shared encoder)", shared, "#0072B2", "-"),
    ("IPPO (separate networks)", indp, "#D55E00", (0, (5, 1.5))),
]
for lbl, r, _, _ in RQ1:
    n_eps = pd.to_numeric(r["history"][X], errors="coerce").max()
    print(f"RQ1 {lbl:28s} episodes={n_eps:.0f}  runtime={r['runtime_s']/3600:.2f} h")

fig, ax = plt.subplots(figsize=(3.3, 2.4))
for lbl, r, color, ls in RQ1:
    agg = mf.aggregate([r], REW, X, smooth=SMOOTH)
    grid, mean, _, _ = agg
    ax.plot(grid, mean, color=color, ls=ls, lw=1.6, label=lbl)
ax.set_xlabel("Episodes")
ax.set_ylabel("Collective reward")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, handlelength=1.9, loc="lower right")
save(fig, "latex/figures/fig2_reward")

fig, ax = plt.subplots(figsize=(3.3, 2.4))
hours = [r["runtime_s"] / 3600 for _, r, _, _ in RQ1]
ax.bar(range(len(RQ1)), hours, width=0.6,
       color=[c for _, _, c, _ in RQ1], edgecolor="black", linewidth=0.6)
ax.set_xticks(range(len(RQ1)))
ax.set_xticklabels(["Shared\nencoder", "Separate\nnetworks"])
ax.set_ylabel("Runtime (hours)")
ax.set_ylim(0, max(hours) * 1.18)
ax.grid(True, axis="y", alpha=0.3)
for i, h in enumerate(hours):
    ax.text(i, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
save(fig, "latex/figures/fig2_runtime")

# ---------------- fig3: RQ2 IPPO vs MAPPO, reward+entropy, both envs (2x2) ----
groups = {env: load_groups(env) for env, _ in ENVS}
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
for i, (env, name) in enumerate(ENVS):
    draw(axes[i, 0], groups[env], ["IPPO", "MAPPO"], REW, "Collective reward",
         legend=(i == 0), title=name)
    draw(axes[i, 1], groups[env], ["IPPO", "MAPPO"], ENT, "Policy entropy", title=name)
save(fig, "latex/figures/fig3", pad=0.5)

# ---------------- fig4: RQ4 GRPO normalization ablation, both envs (2x2) ------
# columns = env, rows = (reward, entropy) so the env contrast reads left-right
NORM = ["IPPO", "GRPO (step group-norm)", "GRPO (global-norm)"]
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
for i, (env, name) in enumerate(ENVS):
    draw(axes[0, i], groups[env], NORM, REW, "Collective reward",
         legend=(i == 0), title=name)
    draw(axes[1, i], groups[env], NORM, ENT, "Policy entropy", title=name)
save(fig, "latex/figures/fig4_norm", pad=0.5)

# ---------------- printed stats for the text ---------------------------------
vram = {}
for env, _ in ENVS:
    for lbl, runs in groups[env].items():
        fam = "GRPO" if lbl.startswith("GRPO") else lbl
        for e in runs:
            col = pd.to_numeric(e["history"].get("system/peak_vram_gb"), errors="coerce").dropna()
            if len(col):
                vram.setdefault(fam, []).append(float(col.mean()))
ppo = np.mean(vram["IPPO"] + vram["MAPPO"])
grpo = np.mean(vram["GRPO"])
print(f"\nPeak VRAM: PPO-family {ppo:.3f} GB vs GRPO {grpo:.3f} GB "
      f"-> {ppo/grpo:.2f}x, {(1 - grpo/ppo)*100:.1f}% reduction")
for env, name in ENVS:
    for lbl in ("IPPO", "MAPPO", "GRPO (step group-norm)", "GRPO (global-norm)"):
        agg = mf.aggregate(groups[env].get(lbl, []), REW, X, smooth=0.0)
        if agg:
            grid, mean, sem, n = agg
            k = max(1, int(0.1 * len(grid)))  # mean over final 10% of training
            print(f"{name:10s} {lbl:24s} final reward {mean[-k:].mean():.2f} "
                  f"+- {sem[-k:].mean():.2f} (n={n})")
