#!/usr/bin/env python3
"""Fetch IGRPO wandb runs, select the most-recent successful run per (variant, seed),
aggregate across seeds, and plot paper-style figures.

Target variants (everything else in the project is ignored):
    IPPO, MAPPO, GRPO (step-norm), GRPO (group-norm)
  - "step-norm"  = per-timestep cross-agent stats  (global_norm=False)
  - "group-norm" = single global mean/std over agents & time (global_norm=True)

Selection: list run metadata first (cheap), pick the latest *finished* run per
(variant, seed), and only then download history for those ~12 runs -- so the many
failed/old runs are never streamed.

Outputs are per-environment so different envs never clobber each other:
    figures/<tag>/...                       # all panels + cache + aggregated.csv
    latex/figures/<tag>_reward.* / <tag>_vram.*   # paper-ready copies

Examples
--------
    python3 make_figures.py                 # fetch (cached) + plot default project
    python3 make_figures.py --project multigrid-ippo-MultiGrid-Cluttered-Fixed-15x15
    python3 make_figures.py --refresh       # re-pull from wandb, overwrite cache
    python3 make_figures.py --x step/x_axis # x-axis = env steps instead of episodes
    python3 make_figures.py --smooth 0.0    # disable EMA smoothing

Auth: ~/.netrc (run `python3 -m wandb login` in a real terminal if needed).
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "ko120"
PROJECT = "multigrid-ippo-MultiGrid-FourRooms-15x15-v0"
FIGROOT = "figures"


def env_tag(project):
    """Short env slug from a wandb project name, used for output dirs/filenames.

    'multigrid-ippo-MultiGrid-Cluttered-Fixed-15x15'  -> 'cluttered'
    'multigrid-ippo-MultiGrid-FourRooms-15x15-v0'     -> 'fourrooms'
    """
    t = project
    for pre in ("multigrid-ippo-", "MultiGrid-"):
        if t.startswith(pre):
            t = t[len(pre):]
    return (t.split("-")[0] or project).lower()

# (wandb key, human-readable axis label). VRAM panel auto-dropped if absent.
METRICS = [
    ("episode/collective_reward_mean", "Collective reward"),
    ("train/entropy", "Policy entropy"),
    ("episode/episode_length_mean", "Episode length"),
    ("train/policy_loss", "Policy loss"),
    ("train/kl_loss", "KL loss"),
    # ("system/peak_vram_gb", "Peak VRAM (GB)"),  # skipped for now: logged ~1.2x, not the 8x claim
]

# Fixed legend/colour order for the paper.
ORDER = ["IPPO", "MAPPO", "GRPO (step group-norm)", "GRPO (global-norm)"]

# Colourblind-safe palette (Okabe-Ito) + distinct linestyles per variant.
PALETTE = {
    "IPPO": "#0072B2",                       # blue
    "MAPPO": "#E69F00",                      # orange
    "GRPO (step group-norm)": "#009E73",     # bluish green
    "GRPO (global-norm)": "#D55E00",         # vermillion
}
LINESTYLE = {
    "IPPO": "-",
    "MAPPO": (0, (5, 1.5)),                  # dashed
    "GRPO (step group-norm)": (0, (1, 1)),   # dotted
    "GRPO (global-norm)": (0, (4, 1, 1, 1)), # dash-dot
}
_FALLBACK = ["#CC79A7", "#56B4E9", "#F0E442", "#000000"]  # extra Okabe-Ito hues

# Paper-ready style: serif (~Computer Modern), sized for 0.49*textwidth (~2.7in).
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
})

# Paper-ready copies land in latex/figures/<tag>_reward.* and <tag>_vram.*
# (main.tex sets \graphicspath{{figures/}}, so figures live in latex/figures/).
# Reference them from main.tex with \includegraphics{<tag>_reward.pdf} etc.
LATEX_DIR = os.path.join("latex", "figures")


def variant_label(cfg):
    """Map a run's logged config to a canonical paper label, or None to ignore it."""
    mode = str(cfg.get("mode", "")).lower()
    if mode == "grpo":
        if cfg.get("backward"):
            return None  # experimental, not a paper variant
        if not bool(cfg.get("use_return", True)):  # code default use_return=True
            return None  # raw-reward GRPO not in the paper's 4-way comparison
        gn = bool(cfg.get("global_norm", False))   # code default global_norm=False
        return "GRPO (global-norm)" if gn else "GRPO (step group-norm)"
    if mode == "ppo":
        algo = str(cfg.get("algorithm", "IPPO") or "IPPO").upper()
        return algo if algo in ("IPPO", "MAPPO") else None
    return None


def fetch_meta(entity, project):
    """List run metadata (no history); carry the run handle for deferred loading."""
    import wandb

    api = wandb.Api(timeout=60)
    runs = api.runs(f"{entity}/{project}")
    metas = []
    for run in runs:
        cfg = dict(run.config)
        metas.append(
            {
                "name": run.name,
                "cfg": cfg,
                "seed": cfg.get("seed"),
                "created_at": str(run.created_at),
                "state": run.state,
                "label": variant_label(cfg),
                "_run": run,
            }
        )
    print(f"Listed {len(metas)} runs in {entity}/{project}")
    return metas


def select_and_load(metas):
    """Pick latest finished run per (label, seed) and download its history only."""
    buckets = defaultdict(list)
    for m in metas:
        if m["label"] is not None:
            buckets[(m["label"], m["seed"])].append(m)

    selected, dropped = [], []
    for (lbl, seed), cands in buckets.items():
        # newest first, then finished-before-unfinished (stable sort preserves recency)
        cands = sorted(cands, key=lambda m: m["created_at"], reverse=True)
        cands = sorted(cands, key=lambda m: 0 if m["state"] == "finished" else 1)
        chosen = None
        for m in cands:
            try:
                hist = pd.DataFrame(m["_run"].scan_history())
            except Exception as e:  # noqa: BLE001
                print(f"  scan_history failed for {m['name']!r}: {e}")
                hist = pd.DataFrame()
            if hist.empty:
                dropped.append({**{k: m[k] for k in ("name", "seed", "state", "created_at", "label")},
                                "why": "empty"})
                continue
            m = {k: m[k] for k in ("name", "cfg", "seed", "created_at", "state", "label")}
            m["history"] = hist
            chosen = m
            break
        if chosen is None:
            continue
        selected.append(chosen)
        for m in cands:
            if m["name"] != chosen["name"]:
                dropped.append({**{k: m[k] for k in ("name", "seed", "state", "created_at", "label")},
                                "why": "superseded"})
    return selected, dropped


def ema(y, alpha):
    if alpha <= 0:
        return np.asarray(y, dtype=float)
    out = np.empty(len(y), dtype=float)
    acc = float(y[0])
    for i, v in enumerate(y):
        acc = alpha * acc + (1 - alpha) * float(v)
        out[i] = acc
    return out


def aggregate(group, metric, x_key, n_grid=300, smooth=0.0):
    """Interpolate each seed onto a shared grid; return (grid, mean, sem, n)."""
    series = []
    for run in group:
        h = run["history"]
        if metric not in h.columns or x_key not in h.columns:
            continue
        sub = h[[x_key, metric]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 2:
            continue
        x = sub[x_key].to_numpy(float)
        y = sub[metric].to_numpy(float)
        ux, idx = np.unique(x, return_inverse=True)  # collapse duplicate x by mean
        uy = np.zeros_like(ux)
        np.add.at(uy, idx, y)
        uy /= np.bincount(idx)
        series.append((ux, uy))
    if not series:
        return None
    lo = max(s[0][0] for s in series)
    hi = min(s[0][-1] for s in series)
    if not (hi > lo):  # no common overlap -> union range
        lo = min(s[0][0] for s in series)
        hi = max(s[0][-1] for s in series)
    grid = np.linspace(lo, hi, n_grid)
    ys = np.vstack([ema(np.interp(grid, x, y), smooth) for x, y in series])
    mean = ys.mean(axis=0)
    sem = (ys.std(axis=0, ddof=1) / np.sqrt(ys.shape[0])) if ys.shape[0] > 1 else np.zeros_like(mean)
    return grid, mean, sem, ys.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--x", default="episode/x_axis",
                    help="x-axis key: episode/x_axis (episodes) or step/x_axis (env steps)")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA smoothing factor in [0,1); 0 disables")
    ap.add_argument("--refresh", action="store_true", help="re-pull from wandb")
    ap.add_argument("--tag", default=None,
                    help="short env tag for output dir/filenames (default: from --project)")
    args = ap.parse_args()

    tag = args.tag or env_tag(args.project)
    outdir = os.path.join(FIGROOT, tag)
    cache = os.path.join(outdir, "raw_history.pkl")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(LATEX_DIR, exist_ok=True)
    print(f"Project {args.entity}/{args.project} -> {outdir}/ (latex stems '{tag}_*')")

    if args.refresh or not os.path.exists(cache):
        metas = fetch_meta(args.entity, args.project)
        selected, dropped = select_and_load(metas)
        with open(cache, "wb") as f:
            pickle.dump({"selected": selected, "dropped": dropped}, f)
        print(f"Cached selection -> {cache}")
    else:
        with open(cache, "rb") as f:
            blob = pickle.load(f)
        selected, dropped = blob["selected"], blob["dropped"]
        print(f"Loaded {len(selected)} runs from cache {cache} (use --refresh to re-pull)")

    groups = defaultdict(list)
    for e in selected:
        e["label"] = variant_label(e["cfg"])  # recompute so renames don't need --refresh
        groups[e["label"]].append(e)
    labels = [l for l in ORDER if l in groups] + [l for l in groups if l not in ORDER]

    print(f"\nSelected {len(selected)} runs (latest successful per variant x seed); "
          f"dropped {len(dropped)} older/failed/empty runs.")
    print(f"{'variant':18s} {'seed':>5s}  {'state':9s} {'rows':>6s}  created_at            run")
    for lbl in labels:
        for e in sorted(groups[lbl], key=lambda e: str(e["seed"])):
            print(f"{lbl:18s} {str(e['seed']):>5s}  {e['state']:9s} "
                  f"{len(e['history']):>6d}  {e['created_at']:21s} {e['name']}")
    print("\nSeed counts:")
    for lbl in labels:
        print(f"  {lbl:18s} n={len(groups[lbl])}")

    colors = {lbl: PALETTE.get(lbl) for lbl in labels}
    for j, lbl in enumerate([l for l in labels if colors.get(l) is None]):
        colors[lbl] = _FALLBACK[j % len(_FALLBACK)]
    styles = {lbl: LINESTYLE.get(lbl, "-") for lbl in labels}
    xlabel = "Environment steps" if args.x.startswith("step") else "Episodes"

    present = set().union(*[set(e["history"].columns) for e in selected]) if selected else set()
    metrics = [(k, t) for k, t in METRICS if k in present]
    dropped_m = [t for k, t in METRICS if k not in present]
    if dropped_m:
        print(f"\nNo data for: {', '.join(dropped_m)} (panel skipped)")

    csv_rows = []

    def draw(ax, key, title, legend=True):
        drew = False
        for lbl in labels:
            agg = aggregate(groups[lbl], key, args.x, smooth=args.smooth)
            if agg is None:
                continue
            grid, mean, sem, n = agg
            ax.plot(grid, mean, color=colors[lbl], ls=styles[lbl], lw=1.6, label=lbl)
            ax.fill_between(grid, mean - sem, mean + sem, color=colors[lbl], alpha=0.18, lw=0)
            drew = True
            for gx, gm, gs in zip(grid, mean, sem):
                csv_rows.append({"metric": key, "variant": lbl, "x": gx,
                                 "mean": gm, "sem": gs, "n_seeds": n})
        ax.set_xlabel(xlabel)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        if legend and drew:
            ax.legend(frameon=False, handlelength=1.9)
        return drew

    # ---- individual figures (paper-sized: ~0.49*textwidth) ----
    latex_copy = {"episode/collective_reward_mean": f"{tag}_reward"}
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        if draw(ax, key, title):
            fig.tight_layout(pad=0.3)
            stem = os.path.join(outdir, key.replace("/", "_"))
            fig.savefig(stem + ".pdf")
            fig.savefig(stem + ".png", dpi=300)
            print(f"  wrote {stem}.pdf / .png")
            if key in latex_copy:
                tstem = os.path.join(LATEX_DIR, latex_copy[key])
                fig.savefig(tstem + ".pdf")
                fig.savefig(tstem + ".png", dpi=300)
                print(f"  wrote {tstem}.pdf / .png  (paper slot)")
        plt.close(fig)

    # ---- VRAM as bars -> teaser right half (latex/fig1_vram.*) ----
    VRAM = "system/peak_vram_gb"
    vram_vals = {}
    for lbl in labels:
        vals = []
        for e in groups[lbl]:
            if VRAM in e["history"].columns:
                col = pd.to_numeric(e["history"][VRAM], errors="coerce").dropna()
                if len(col):
                    vals.append(float(col.mean()))
        if vals:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            vram_vals[lbl] = (float(np.mean(vals)), sem)

    SHORT = {"IPPO": "IPPO", "MAPPO": "MAPPO",
             "GRPO (step group-norm)": "GRPO\n(step)", "GRPO (global-norm)": "GRPO\n(global)"}

    def draw_vram_bar(ax):
        bl = [l for l in labels if l in vram_vals]
        means = [vram_vals[l][0] for l in bl]
        sems = [vram_vals[l][1] for l in bl]
        ax.bar(range(len(bl)), means, yerr=sems, width=0.7, capsize=3,
               color=[colors[l] for l in bl], edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(bl)))
        ax.set_xticklabels([SHORT.get(l, l) for l in bl])
        ax.set_ylabel("Peak VRAM (GB)")
        ax.set_ylim(0, max(means) * 1.18)
        ax.grid(True, axis="y", alpha=0.3)
        for i, m in enumerate(means):
            ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    if vram_vals:
        fig, ax = plt.subplots(figsize=(3.3, 2.4))
        draw_vram_bar(ax)
        fig.tight_layout(pad=0.3)
        for stem in (os.path.join(outdir, "system_peak_vram_gb_bar"),
                     os.path.join(LATEX_DIR, f"{tag}_vram")):
            fig.savefig(stem + ".pdf")
            fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)
        print(f"  wrote {os.path.join(LATEX_DIR, tag + '_vram')}.pdf / .png  (paper slot)")

        # reward + VRAM side-by-side preview (teaser-style layout)
        fig, (axr, axv) = plt.subplots(1, 2, figsize=(6.6, 2.6))
        draw(axr, "episode/collective_reward_mean", "Collective reward", legend=True)
        draw_vram_bar(axv)
        fig.tight_layout(pad=0.5)
        stem = os.path.join(outdir, "reward_vram")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)
        print(f"  wrote {stem}.pdf / .png  (reward+VRAM preview)")

    # ---- bonus: reward + entropy side by side (full textwidth) ----
    rq = [m for m in metrics if m[0] in ("episode/collective_reward_mean", "train/entropy")]
    if len(rq) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
        for i, (ax, (key, title)) in enumerate(zip(axes, rq)):
            draw(ax, key, title, legend=(i == 0))
        fig.tight_layout(pad=0.4)
        stem = os.path.join(outdir, "reward_entropy")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png", dpi=300)
        plt.close(fig)
        print(f"  wrote {stem}.pdf / .png")

    # ---- combined overview ----
    ncol = 2
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.0 * nrow), squeeze=False)
    handles_labels = None
    for ax, (key, title) in zip(axes.flat, metrics):
        draw(ax, key, title, legend=False)
        if handles_labels is None and ax.get_legend_handles_labels()[0]:
            handles_labels = ax.get_legend_handles_labels()
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    if handles_labels:
        fig.legend(*handles_labels, loc="lower center", ncol=len(labels),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    stem = os.path.join(outdir, "combined")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")

    pd.DataFrame(csv_rows).drop_duplicates().to_csv(os.path.join(outdir, "aggregated.csv"), index=False)
    print(f"  wrote {os.path.join(outdir, 'aggregated.csv')}")


if __name__ == "__main__":
    main()
