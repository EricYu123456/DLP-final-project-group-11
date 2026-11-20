"""Fit the ten-source public capability embedding for a user-supplied pool of models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from cover_release_panels import DATA, OUT, SEEDS
from fit_roster_panel import fit_pool
from panel_capabilities import panel_text
from serve_confirmation_router import measurements, write_probes
from threadpoolctl import threadpool_limits

from coldstart.data import BENCHMARKS


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan, fit = sub.add_parser("plan"), sub.add_parser("fit")
    for cmd in (plan, fit):
        cmd.add_argument("--model-name", action="append", default=[])
        cmd.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
        cmd.add_argument("--responses", type=Path, required=cmd is fit)
        cmd.add_argument("--output", type=Path, required=True)
    fit.add_argument(
        "--embeddings", type=Path, help="Also export each model's ability vector as NumPy arrays."
    )
    args = parser.parse_args(argv)
    with np.load(OUT / "panels.npz") as panels, np.load(DATA / "processed/embedllm.npz") as corpus:
        ids = panels["coverage"][SEEDS.index(args.seed)]
        domain = corpus["benchmark_ids"][ids]
    names, y = measurements(args.responses, args.model_name, ids)
    texts = panel_text(ids, DATA / "raw")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        write_probes(args.output, names, ids, [BENCHMARKS[j] for j in domain], texts, y)
    else:
        if not np.isfinite(y).all():
            parser.error("Fill the response for all 200 prompts for every candidate.")
        with threadpool_limits(limits=2):
            pool, _ = fit_pool(None, domain, y, texts, ids, names, args.seed, text_only=True)
        pool["method"] = "release-text-200"
        joblib.dump(pool, args.output, compress=2)
        if args.embeddings:
            args.embeddings.parent.mkdir(parents=True, exist_ok=True)
            skill = pool["skill"]
            axes = [*BENCHMARKS, *(f"latent_{j + 1}" for j in range(skill["axes"].shape[1]))]
            with args.embeddings.open("wb") as f:
                np.savez_compressed(
                    f,
                    embeddings=np.column_stack((skill["z"], skill["axes"])),
                    model_names=pool["model_names"],
                    coordinate_names=axes,
                    probe_ids=ids,
                )
    print(json.dumps({"models": len(names), "questions_per_model": len(ids), "output": str(args.output)}))
    return 0
