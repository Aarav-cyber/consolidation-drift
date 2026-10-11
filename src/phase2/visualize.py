"""Visualization suite for Phase 2 consolidation drift metrics.

Generates:
1. results/figures/budget_depth_heatmap.png - 2D heatmap across budget ladder and round depth
2. results/figures/arc_trajectory.png       - Round-by-round trajectory of ARC slope (k=1..10)
3. results/figures/arc_scatter_ols.png      - OLS scatter plots (x: true_level - 3, y: drift)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(drift_metrics, out_dir):
    series = drift_metrics.get("series", {})
    arc_by_round = drift_metrics.get("arc_by_round", {})
    
    pipelines = ["sliding", "hierarchical"]
    levels = ["c0", "c8", "c20"]
    rounds = list(range(1, 11))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Consolidation Drift: Budget × Depth Grid (Headline Comparison)", fontsize=16, fontweight="bold")
    
    # 1. Mean signed drift heatmaps
    for idx, pipe in enumerate(pipelines):
        ax = axes[0, idx]
        grid = np.zeros((len(levels), len(rounds)))
        for r_idx, lvl in enumerate(levels):
            # aggregate across categories for this pipe and lvl
            for c_idx, k in enumerate(rounds):
                # gather mean_signed for all categories
                drifts = []
                for key, val in series.items():
                    if val["pipeline"] == pipe and val["context_level"] == lvl:
                        r_data = val["rounds"].get(str(k))
                        if r_data:
                            drifts.append(r_data["mean_signed"])
                grid[r_idx, c_idx] = np.mean(drifts) if drifts else 0.0
                
        im = ax.imshow(grid, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_title(f"{pipe.capitalize()} Pipeline: Mean Signed Drift", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels)
        ax.set_xlabel("Consolidation Depth (Round k)")
        ax.set_ylabel("Memory Budget / Context Level")
        
        # Annotate numbers
        for i in range(len(levels)):
            for j in range(len(rounds)):
                val = grid[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        color="white" if abs(val) > 0.8 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. ARC Slope heatmaps
    for idx, pipe in enumerate(pipelines):
        ax = axes[1, idx]
        grid = np.zeros((len(levels), len(rounds)))
        for r_idx, lvl in enumerate(levels):
            for c_idx, k in enumerate(rounds):
                key = f"{pipe}|{lvl}|{k}"
                arc = arc_by_round.get(key, {})
                grid[r_idx, c_idx] = arc.get("slope", 0.0)
                
        im = ax.imshow(grid, cmap="RdYlBu", vmin=-1.2, vmax=0.4, aspect="auto")
        ax.set_title(f"{pipe.capitalize()} Pipeline: Attractor Regression Coeff (ARC)", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels)
        ax.set_xlabel("Consolidation Depth (Round k)")
        ax.set_ylabel("Memory Budget / Context Level")
        
        for i in range(len(levels)):
            for j in range(len(rounds)):
                val = grid[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        color="white" if val < -0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = out_dir / "budget_depth_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_arc_trajectories(drift_metrics, out_dir):
    arc_by_round = drift_metrics.get("arc_by_round", {})
    rounds = list(range(1, 11))
    
    plt.figure(figsize=(10, 6))
    styles = {
        ("sliding", "c0"): ("#2ca02c", "--", "s", "sliding | c0 (no-op control)"),
        ("sliding", "c8"): ("#1f77b4", "-.", "o", "sliding | c8"),
        ("sliding", "c20"): ("#17becf", "-", "^", "sliding | c20"),
        ("hierarchical", "c0"): ("#ff7f0e", "--", "d", "hierarchical | c0"),
        ("hierarchical", "c8"): ("#d62728", "-.", "v", "hierarchical | c8"),
        ("hierarchical", "c20"): ("#9467bd", "-", "X", "hierarchical | c20 (fusion artifact)"),
    }
    
    for (pipe, lvl), (col, ls, marker, label) in styles.items():
        slopes = [arc_by_round.get(f"{pipe}|{lvl}|{k}", {}).get("slope", 0.0) for k in rounds]
        plt.plot(rounds, slopes, color=col, linestyle=ls, marker=marker, linewidth=2, label=label)
        
    plt.axhline(0.0, color="gray", linestyle=":", linewidth=1.5, label="Unbiased Null (ARC = 0)")
    plt.axhline(-1.0, color="darkred", linestyle=":", linewidth=1.5, label="Complete Collapse (ARC = -1)")
    
    plt.title("Attractor Regression Coefficient (ARC) Trajectory Across Rounds", fontsize=14, fontweight="bold")
    plt.xlabel("Consolidation Round (k)", fontsize=12)
    plt.ylabel("ARC Slope (b in drift ~ b · (true_level - 3))", fontsize=12)
    plt.xticks(rounds)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
    plt.tight_layout()
    
    out_path = out_dir / "arc_trajectory.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter_ols(evaluations, out_dir):
    # Scatter of x = true_level - 3 vs y = drift
    configs = [
        ("sliding", "c0"), ("sliding", "c8"), ("sliding", "c20"),
        ("hierarchical", "c0"), ("hierarchical", "c8"), ("hierarchical", "c20")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (pipe, lvl) in enumerate(configs):
        ax = axes[idx]
        rows = [r for r in evaluations if r.get("pipeline") == pipe and r.get("context_level") == lvl
                and r.get("level") is not None and r.get("true_level") is not None]
        
        xs = np.array([r["true_level"] - 3 for r in rows])
        ys = np.array([r["level"] - r["true_level"] for r in rows])
        
        # Add jitter for visualization
        jitter_x = xs + np.random.normal(0, 0.05, size=len(xs))
        jitter_y = ys + np.random.normal(0, 0.05, size=len(ys))
        
        ax.scatter(jitter_x, jitter_y, alpha=0.35, color="#1f77b4", edgecolor="none", s=30)
        
        # Fit OLS
        if len(xs) > 1 and np.var(xs) > 0:
            slope, intercept = np.polyfit(xs, ys, 1)
            line_x = np.linspace(-2, 2, 100)
            line_y = intercept + slope * line_x
            ax.plot(line_x, line_y, color="red", linewidth=2, label=f"Fit: b={slope:+.3f}")
        else:
            slope, intercept = 0.0, 0.0
            ax.plot([-2, 2], [0, 0], color="red", linewidth=2, label="Fit: b=0.000")
            
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{pipe} | {lvl}\nARC slope={slope:+.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("x = true_level − 3")
        ax.set_ylabel("y = level − true_level (drift)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-3.5, 3.5)
        
    plt.suptitle("ARC OLS Regression Fit per Pipeline & Context Level (Pooled)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "arc_scatter_ols.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    exp_dir = Path("data/experiments/phase2")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    drift_metrics = json.loads((exp_dir / "drift_metrics.json").read_text(encoding="utf-8"))
    evaluations = json.loads((exp_dir / "evaluations.json").read_text(encoding="utf-8"))
    
    plot_heatmaps(drift_metrics, out_dir)
    plot_arc_trajectories(drift_metrics, out_dir)
    plot_scatter_ols(evaluations, out_dir)


if __name__ == "__main__":
    main()
