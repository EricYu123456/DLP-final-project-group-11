"""Report matched family experiments without pooling their distinct protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


LABELS = {"coldstart_structured8": "Structured embedding (14D)",
          "coldstart_rank0": "Global + benchmark (6D)",
          "training_domain_mean": "Training benchmark mean",
          "domain_probe_mean": "Smoothed benchmark mean", "question_knn": "Question KNN",
          "global_probe_mean": "Smoothed global mean"}
FAMILIES = ["Llama", "Mistral", "Yi"]
ARMS = ["excluded", "related_allowed"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed_path = args.run / "test_completed.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if not completed.get("static_probe_ids_matched_within_pairs"):
        raise ValueError("The experiment has not verified matched probe IDs")
    if set(completed["families"]) != set(FAMILIES):
        raise ValueError("Report the three predeclared family cases together")
    source = args.run / "family_per_model_metrics.csv"
    rows = pd.read_csv(source)
    key = ["family_case", "condition", "predictor", "strategy", "budget", "seed", "model_id"]
    if rows.duplicated(key).any():
        raise ValueError("Duplicate family experiment cells")
    if set(rows.condition) != set(ARMS) or set(rows.family_case) != set(FAMILIES):
        raise ValueError("Both matched conditions are required for every family")
    if "phase" in rows and set(rows.phase) != {"test"}:
        raise ValueError("Family report expects completed test results only")
    paired_keys = ["family_case", "predictor", "strategy", "budget", "seed", "model_id"]
    excluded = rows[rows.condition == "excluded"]
    related = rows[rows.condition == "related_allowed"]
    paired = excluded.merge(related, on=paired_keys, suffixes=("_excluded", "_related"), validate="one_to_one")
    if len(paired) != len(excluded) or len(paired) != len(related):
        raise ValueError("Unmatched family target/model/seed cells")
    paired["macro_mae_gap"] = paired.macro_mae_excluded - paired.macro_mae_related
    paired["brier_gap"] = paired.brier_excluded - paired.brier_related
    per_model_keys = ["family_case", "predictor", "strategy", "budget", "model_id"]
    model_differences = paired.groupby(per_model_keys, as_index=False)[["macro_mae_gap", "brier_gap"]].mean()
    summaries = []
    summary_keys = ["family_case", "predictor", "strategy", "budget"]
    for name, group in model_differences.groupby(summary_keys):
        original = paired
        for column, value in zip(summary_keys, name):
            original = original[original[column] == value]
        summaries.append({**dict(zip(summary_keys, name)),
                          "macro_mae_gap": float(group.macro_mae_gap.mean()),
                          "brier_gap": float(group.brier_gap.mean()),
                          "models_with_positive_gap": int((group.macro_mae_gap > 0).sum()),
                          "n_models": len(group), "n_seeds": int(original.seed.nunique()),
                          "direction": "excluded minus related_allowed; positive means relatives helped",
                          "confidence_interval": "suppressed: one lineage block per case"})
    differences = pd.DataFrame(summaries)
    condition_keys = ["family_case", "condition", "predictor", "strategy", "budget", "model_id"]
    model_means = rows.groupby(condition_keys, as_index=False)[["macro_mae", "brier"]].mean()
    condition_summary = model_means.groupby(condition_keys[:-1], as_index=False)[["macro_mae", "brier"]].mean()
    args.output.mkdir(parents=True, exist_ok=True)
    differences.to_csv(args.output / "paired_summary.csv", index=False)
    model_differences.to_csv(args.output / "paired_per_model.csv", index=False)
    condition_summary.to_csv(args.output / "condition_summary.csv", index=False)
    rows.to_csv(args.output / "per_model_metrics.csv", index=False)
    # Report both estimands rather than silently giving the larger case more weight.
    combined = differences.groupby(["predictor", "strategy", "budget"], as_index=False)[["macro_mae_gap", "brier_gap"]].mean()
    combined["aggregation"] = "equal weight per family case; descriptive across three cases"
    combined.to_csv(args.output / "equal_family_summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    colors = {"coldstart_structured8": "#0072B2", "coldstart_rank0": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), sharey=True)
    for family, ax in zip(FAMILIES, axes):
        subset = condition_summary[(condition_summary.family_case == family)
                                   & (condition_summary.strategy == "balanced_random")]
        for predictor, color in colors.items():
            for arm, style in zip(ARMS, ["--", "-"]):
                group = subset[(subset.predictor == predictor) & (subset.condition == arm)].sort_values("budget")
                if group.empty:
                    raise ValueError(f"Missing predeclared neural comparison: {family}/{predictor}/{arm}")
                ax.plot(group.budget, group.macro_mae * 100, marker="o", color=color, linestyle=style)
        n_models = model_means[model_means.family_case == family].model_id.nunique()
        ax.set(title=f"{family} · {n_models} held-out models", xlabel="Responses per model")
        ax.set_xticks([50, 100, 200])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Macro benchmark MAE (percentage points)")
    handles = [Line2D([0], [0], color=color, label=LABELS[name]) for name, color in colors.items()]
    handles += [Line2D([0], [0], color="#555555", linestyle=style, label=name)
                for style, name in [("--", "Family excluded"), ("-", "Relatives allowed")]]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.04), fontsize=9)
    fig.suptitle("Matched family study: same targets, probes, and 40 training models", y=1.02)
    fig.tight_layout()
    save(fig, args.output / "matched_family_curves")
    max_budget = int(rows.budget.max())
    latest = differences[(differences.strategy == "balanced_random") & (differences.budget == max_budget)
                         & differences.predictor.isin(colors)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(FAMILIES))
    for i, (predictor, color) in enumerate(colors.items()):
        selected = latest[latest.predictor == predictor].set_index("family_case").reindex(FAMILIES)
        ax.bar(x + (i - .5) * .32, selected.macro_mae_gap * 100, width=.3, color=color,
               label=LABELS[predictor])
    ax.axhline(0, color="#444444", linewidth=.9)
    ax.set_xticks(x, FAMILIES)
    ax.set(ylabel="MAE: excluded − relatives allowed (pp)",
           title=f"Effect of replacing relatives with unrelated models\n{max_budget} identical balanced probes; descriptive case results")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=9)
    save(fig, args.output / "family_composition_gap")
    table = latest.copy()
    table["gap_pp"] = table.macro_mae_gap * 100
    lines = ["# Matched family study", "",
             "Both conditions use the same held-out models, Qwen validation models, question groups,",
             "probe IDs, and training-pool size (40). The treatment changes which historical models",
             "fill part of that pool. Positive gaps mean allowing relatives reduced MAE.", "",
             "The fixed family configuration differs from the validation-selected main configuration.",
             "These are within-case comparisons. They do not identify a causal ancestry effect.", "",
             "![Matched curves](matched_family_curves.png)", "",
             "![Composition gap](family_composition_gap.png)", "",
             f"## Results at {max_budget} probes", "",
             "| Family | Predictor | MAE gap (pp) | Models | Seeds |",
             "| --- | --- | ---: | ---: | ---: |"]
    for _, row in table.iterrows():
        lines.append(f"| {row.family_case} | {LABELS[row.predictor]} | {row.gap_pp:.3f} | {row.n_models} | {row.n_seeds} |")
    lines += ["", "Seeds are averaged within each target model. Single-family confidence intervals",
              "are suppressed; combined equal-family summaries across three cases are descriptive.", ""]
    (args.output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    provenance = {"source_run": str(args.run), "source_sha256": {source.name: digest(source)},
                  "study_sha256": completed["study_sha256"],
                  "estimand": "paired excluded minus related-allowed score error; same target models",
                  "intervals": "not reported for individual lineage cases"}
    (args.output / "report_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved matched family report to {args.output}")


if __name__ == "__main__":
    main()
