"""Build reproducible tables and publication figures from saved experiment rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import cycle
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coldstart.metrics import paired_bootstrap


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#666666"]
LABELS = {"structured": "Structured embedding", "structured_rank8": "Structured embedding",
          "coldstart_structured8": "Structured embedding (14D)",
          "coldstart_rank0": "Global + benchmark (6D)",
          "coldstart_neural_lowrank": "Neural low-rank (8D)",
          "scalar_logit_shift": "Scalar calibrated ability (1D)",
          "domain_probe_mean": "Smoothed benchmark mean", "global_probe_mean": "Smoothed global mean",
          "training_domain_mean": "Training benchmark mean (0 probes)",
          "rank0": "Global + benchmark", "structured_rank0": "Global + benchmark",
          "lowrank": "Neural low-rank", "lowrank_rank8": "Neural low-rank",
          "domain_mean": "Smoothed benchmark mean", "smoothed_mean": "Smoothed benchmark mean",
          "question_knn": "Question KNN", "knn": "Question KNN",
          "train_mean": "Training benchmark mean", "training_mean": "Training benchmark mean",
          "random": "Random", "balanced_random": "Balanced random",
          "diversity": "Semantic diversity", "balanced_diversity": "Balanced diversity",
          "fisher": "Score-aware Fisher", "balanced_fisher": "Balanced score-aware Fisher"}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values: pd.Series, model_ids: pd.Series, families: dict[int, str]) -> dict:
    groups = np.array([families.get(int(model), f"model_{model}") for model in model_ids])
    result = {"mean": float(values.mean()), "n_models": len(values),
              "n_groups": len(np.unique(groups)), "ci_low": None, "ci_high": None,
              "interval_unit": "lineage group"}
    if len(np.unique(groups)) >= 2:
        interval = paired_bootstrap(values.to_numpy(), groups=groups, n_bootstrap=10000, seed=8675309)
        result.update(ci_low=interval["ci_low"], ci_high=interval["ci_high"])
    return result


def aggregate(rows: pd.DataFrame, families: dict[int, str]) -> pd.DataFrame:
    metrics = [column for column in ("macro_mae", "worst_domain_mae", "brier", "macro_brier", "nll")
               if column in rows]
    keys = ["predictor", "strategy", "budget"]
    records = []
    for key, group in rows.groupby(keys, dropna=False, sort=True):
        per_model = group.groupby("model_id")[metrics].mean().reset_index()
        for metric in metrics:
            record = dict(zip(keys, key))
            record.update(metric=metric, n_seeds=int(group.seed.nunique()))
            record.update(estimate(per_model[metric], per_model.model_id, families))
            records.append(record)
    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(summary: pd.DataFrame, out: Path, phase: str, predictor: str) -> None:
    selected = summary[(summary.metric == "macro_mae") & (summary.strategy == "balanced_random")]
    selected = selected[~selected.predictor.str.endswith("_permuted") & (selected.predictor != "global_probe_mean")]
    if not selected.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for color, (method, group) in zip(cycle(COLORS), selected.groupby("predictor", sort=True)):
            group = group.sort_values("budget")
            x, y = group.budget.to_numpy(), group["mean"].to_numpy() * 100
            ax.plot(x, y, marker="o", color=color, label=label(method), linewidth=2)
            valid = group.ci_low.notna() & group.ci_high.notna()
            if valid.all():
                ax.fill_between(x, group.ci_low.to_numpy(dtype=float) * 100,
                                group.ci_high.to_numpy(dtype=float) * 100, alpha=.1, color=color)
        zero = summary[(summary.metric == "macro_mae") & (summary.strategy == "zero_probe")]
        if len(zero) == 1:
            ax.axhline(float(zero.iloc[0]["mean"]) * 100, color="#666666", linestyle="--",
                       label="Training benchmark mean (0 probes)")
        ax.set(xlabel="Responses observed per new model", ylabel="Macro benchmark MAE (percentage points)",
               title=f"Model representation comparison · {phase}\nIdentical balanced random probes")
        ax.set_xticks(sorted(selected.budget.unique()))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9, loc="upper right")
        save_figure(fig, out / "embedding_budget_curve")
    selected = summary[(summary.metric == "macro_mae") & (summary.predictor == predictor)]
    if not selected.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for color, (strategy, group) in zip(cycle(COLORS), selected.groupby("strategy", sort=True)):
            group = group.sort_values("budget")
            ax.plot(group.budget, group["mean"] * 100, marker="o", color=color,
                    label=label(strategy), linewidth=2)
        ax.set(xlabel="Responses observed per new model", ylabel="Macro benchmark MAE (percentage points)",
               title=f"Probe policy comparison · {phase}\n{label(predictor)}")
        ax.set_xticks(sorted(selected.budget.unique()))
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=9, loc="upper right")
        save_figure(fig, out / "probe_policy_curve")


def plot_domain_errors(rows: pd.DataFrame, out: Path, phase: str) -> None:
    selected = rows[(rows.strategy == "balanced_random") & (rows.budget == rows.budget.max())].copy()
    selected = selected[~selected.predictor.str.endswith("_permuted") & (selected.predictor != "global_probe_mean")]
    if selected.empty:
        return
    error_column = "error" if "error" in selected else "absolute_error"
    if error_column not in selected:
        selected[error_column] = np.abs(selected.pred_score - selected.true_score)
    model_means = selected.groupby(["predictor", "model_id", "benchmark"])[error_column].mean()
    table = model_means.groupby(["predictor", "benchmark"]).mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(max(7.4, .9 * len(table.columns) + 2.4), max(3.2, .55 * len(table) + 1.5)))
    im = ax.imshow(table.to_numpy(), cmap="YlOrRd", vmin=0, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), table.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(table.index)), [label(value) for value in table.index])
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Per-benchmark score MAE · {phase}\n{int(selected.budget.max())} balanced random probes")
    fig.colorbar(im, ax=ax, label="Percentage points", shrink=.8)
    save_figure(fig, out / "benchmark_errors")


def paired_comparisons(rows: pd.DataFrame, families: dict[int, str], predictor: str,
                       contrasts: list[list[str]] | None = None) -> pd.DataFrame:
    comparisons = []
    if predictor in set(rows.predictor):
        contrasts = contrasts or [[policy, "balanced_random"] for policy in sorted(set(rows.strategy) - {"balanced_random", "zero_probe"})]
        for policy_a, policy_b in contrasts:
            comparisons.append((predictor, policy_a, predictor, policy_b, "probe_policy"))
        for other in sorted(set(rows.predictor) - {predictor}):
            comparisons.append((predictor, "balanced_random", other, "balanced_random", "representation"))
    records = []
    for a, policy_a, b, policy_b, kind in comparisons:
        for budget in sorted(rows.budget.unique()):
            left = rows[(rows.predictor == a) & (rows.strategy == policy_a) & (rows.budget == budget)]
            right = rows[(rows.predictor == b) & (rows.strategy == policy_b) & (rows.budget == budget)]
            merged = left.merge(right, on=["seed", "model_id"], suffixes=("_a", "_b"), validate="one_to_one")
            if merged.empty:
                continue
            if len(merged) != len(left) or len(merged) != len(right):
                raise ValueError("Paired comparison has unmatched model/seed cells")
            merged["difference"] = merged.macro_mae_a - merged.macro_mae_b
            per_model = merged.groupby("model_id")["difference"].mean().reset_index()
            record = {"comparison_type": kind, "predictor_a": a, "strategy_a": policy_a,
                      "predictor_b": b, "strategy_b": policy_b, "budget": int(budget),
                      "direction": "a_minus_b; negative favors a",
                      "n_paired_seeds": int(merged.seed.nunique()), "n_paired_cells": len(merged)}
            record.update(estimate(per_model.difference, per_model.model_id, families))
            records.append(record)
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    # Avoid an optional tabulate dependency for a reproducible report.
    if frame.empty:
        return "No matching rows.\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory with per_model_metrics.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["test", "validation"], default="test")
    parser.add_argument("--predictor", help="Primary predictor; defaults to the locked experiment config")
    parser.add_argument("--family-table", type=Path, default=ROOT / "data/metadata/model_families.csv")
    args = parser.parse_args()
    effective_config_path = args.run / "effective_config.json"
    effective_config = json.loads(effective_config_path.read_text()) if effective_config_path.exists() else {}
    args.predictor = args.predictor or effective_config.get("evaluation", {}).get("primary_predictor", "coldstart_structured8")
    args.output.mkdir(parents=True, exist_ok=True)
    completion = args.run / f"{args.phase}_completed.json"
    if not completion.exists():
        raise ValueError(f"Refusing to report an incomplete phase: missing {completion.name}")
    model_file = args.run / f"{args.phase}_per_model_metrics.csv"
    domain_file = args.run / f"{args.phase}_per_domain_metrics.csv"
    if args.phase == "test" and not model_file.exists():
        model_file = args.run / "per_model_metrics.csv"
        domain_file = args.run / "per_domain_metrics.csv"
    rows = pd.read_csv(model_file)
    if "phase" in rows:
        accepted = {"test"} if args.phase == "test" else {"validation", "validate", "val"}
        rows = rows[rows.phase.isin(accepted)].copy()
    if rows.empty:
        raise ValueError(f"No {args.phase} results in {model_file}")
    for column in ["protocol", "label_policy", "model_split"]:
        if column in rows and rows[column].nunique() != 1:
            raise ValueError(f"Report one {column} at a time; do not pool different protocols")
    if rows.duplicated(["predictor", "strategy", "budget", "seed", "model_id"]).any():
        raise ValueError("Duplicate experiment cells; select one locked configuration")
    expected_grid = set((int(seed), int(model)) for seed in rows.seed.unique() for model in rows.model_id.unique())
    for key, group in rows.groupby(["predictor", "strategy", "budget"], dropna=False):
        grid = set(zip(group.seed.astype(int), group.model_id.astype(int)))
        if grid != expected_grid:
            raise ValueError(f"Incomplete seed/model grid for {key}")
    if args.predictor not in set(rows.predictor):
        raise ValueError(f"Unknown primary predictor {args.predictor}; available: {sorted(rows.predictor.unique())}")
    families = {}
    if args.family_table.exists():
        family_rows = pd.read_csv(args.family_table).fillna("")
        families = {int(row.model_id): str(row.family) or f"model_{row.model_id}"
                    for _, row in family_rows.iterrows()}
    summary = aggregate(rows, families)
    contrasts = effective_config.get("evaluation", {}).get("paired_comparisons")
    comparisons = paired_comparisons(rows, families, args.predictor, contrasts)
    summary.to_csv(args.output / "summary.csv", index=False)
    comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
    rows.to_csv(args.output / "per_model_metrics.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    plot_curves(summary, args.output, args.phase, args.predictor)
    sources = {"per_model_metrics.csv": sha256(model_file)}
    if domain_file.exists():
        domains = pd.read_csv(domain_file)
        if "phase" in domains:
            domains = domains[domains.phase.isin(accepted)]
        plot_domain_errors(domains, args.output, args.phase)
        domains.to_csv(args.output / "per_domain_metrics.csv", index=False)
        sources["per_domain_metrics.csv"] = sha256(domain_file)
    provenance = {"phase": args.phase, "source_run": str(args.run), "source_sha256": sources,
                  "score_unit": "proportion; figures use percentage points",
                  "aggregation": "average seeds within each model, then average models",
                  "interval": "95% percentile bootstrap of reviewed lineage groups; descriptive with few groups",
                  "primary_predictor": args.predictor}
    (args.output / "report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    table = summary[(summary.metric == "macro_mae") & summary.strategy.isin(["balanced_random", "zero_probe"])].copy()
    for column in ["mean", "ci_low", "ci_high"]:
        table[column] = pd.to_numeric(table[column]) * 100
    lines = [f"# Benchmark prediction results ({args.phase})", "",
             "Errors below are percentage points. Each benchmark has equal weight.",
             "Seeds are averaged within each held-out model before averaging models.",
             "Intervals resample reviewed lineage groups, and are descriptive when few groups remain.", "",
             "## Balanced random probes", "",
             markdown_table(table, ["predictor", "budget", "mean", "ci_low", "ci_high", "n_models", "n_seeds"]),
             "![Representation comparison](embedding_budget_curve.png)", "",
             "![Probe policies](probe_policy_curve.png)", "",
             "![Benchmark errors](benchmark_errors.png)", "",
             "See `paired_comparisons.csv` for paired differences. Negative values favor predictor/policy A.",
             "The saved rows and source hashes allow the report to be regenerated.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
