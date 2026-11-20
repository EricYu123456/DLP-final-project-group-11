"""Find the smallest nested panel that beats published-default routing on validation."""

import argparse
import json

import numpy as np
import pandas as pd

from confirm_release_panel import main as panel
from evaluate_capability_router import ROOT
from run_experiment import atomic_csv, atomic_json


def run(text_kernel=False):
    release = ROOT / "reports/routing/capability/release_confirmation"
    report = release / ("text_kernel_probe_cost" if text_kernel else "probe_cost")
    report.mkdir(exist_ok=True)
    records = []
    full_name, flag = ("text-kernel", "--text-kernel") if text_kernel else ("tabular-panel", "--tabular")
    budgets = range(40, 201, 20) if text_kernel else (40, 80, 120, 160, 200)
    for n in budgets:
        name = full_name if n == 200 else f"{'text' if text_kernel else 'tabular'}-budget-{n}"
        data = release if n == 200 else report
        options = ["--report", str(data), "--name", name, flag, "--probe-count", str(n)]
        if not (data / f"{name}_selection.json").is_file():
            panel(["--phase", "fit", *options])
        if not (data / f"{name}_valid_paired.csv").is_file():
            panel(["--phase", "score", "--part", "valid", *options])
        results = pd.read_csv(data / f"{name}_valid.csv").query("method == @name")
        gain = pd.read_csv(data / f"{name}_valid_paired.csv").query("reference == 'original-noisy'").iloc[0]
        cfg = json.loads((data / f"{name}_selection.json").read_text(encoding="utf-8"))
        records.append(
            dict(
                name=name,
                probes=n,
                fit_seconds=np.mean([v["seconds"] for v in cfg.values()]),
                **results[["macro", "micro", "top_share"]].mean().to_dict(),
                **gain[["micro_ci_low", "macro_ci_low", "micro_gain", "macro_gain"]].to_dict(),
            )
        )
        print("Probe budget", n, records[-1], flush=True)
    full = records[-1]
    eligible = [
        r for r in records if r["micro_ci_low"] > 0 and r["macro_ci_low"] > 0 and r["top_share"] < 0.97
    ]
    shortest = min(eligible, key=lambda r: r["probes"])
    near = min(
        (r for r in eligible if r["micro"] >= full["micro"] - 0.01 and r["macro"] >= full["macro"] - 0.01),
        key=lambda r: r["probes"],
    )
    atomic_csv(report / "validation.csv", records)
    atomic_json(
        report / "selection.json",
        dict(
            shortest=shortest,
            near_200=near,
            rule="Smallest nested panel with positive lower paired macro and micro bounds against default EmbedLLM and concentration below 97% on validation. Near-200 additionally keeps both accuracy metrics within one percentage point of the 200-prompt setting.",
        ),
    )
    for name in dict.fromkeys([shortest["name"], near["name"]]):
        if name == full["name"]:
            continue
        for target in ("reserve", "test"):
            if not (report / f"{name}_{target}_paired.csv").is_file():
                panel(
                    [
                        "--phase",
                        "score",
                        "--report",
                        str(report),
                        "--name",
                        name,
                        flag,
                        "--part",
                        target,
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-kernel", action="store_true")
    run(parser.parse_args().text_kernel)
