"""Time the normal streaming router on raw confirmation prompts and compare its choices with the saved scores."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import joblib
import numpy as np

from embed_new_model import read_prompts
from evaluate_capability_router import ROOT, input_context
from run_experiment import atomic_json


@contextmanager
def sample_gpu(enabled):
    samples = []
    if not enabled:
        yield samples
        return
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            "500",
        ]
        proc = subprocess.Popen(cmd, stdout=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            yield samples
        finally:
            proc.terminate()
            proc.wait()
            log.seek(0)
            for line in log:
                fields = line.strip().split(",")
                if len(fields) == 2 and fields[0].strip().isdigit():
                    samples.append(
                        (int(fields[0]), float(fields[1]) if fields[1].strip() != "[N/A]" else np.nan)
                    )


def measure(args):
    published = args.roster == "published_112"
    if published:
        import pandas as pd
        from confirm_release_panel import serving_method

        out, report = (
            args.out.with_name("release_confirmation"),
            args.report.with_name("release_confirmation"),
        )
        method_report, method = serving_method(report, args.method, out)
        spec = json.loads((method_report / f"{method}_selection.json").read_text(encoding="utf-8"))["11"]
        probes, seed = spec["questions"], 11
        names = pd.read_csv(ROOT / "data/raw/model_order.csv").sort_values("model_id").model_name
        names = names.str.replace("__", "/", regex=False).tolist()
        with np.load(out / "reserve.npz") as data:
            fresh = data["prompt_ids"]
        with np.load(ROOT / "data/processed/embedllm.npz") as data:
            domains = data["benchmark_ids"][fresh]
        path = out / f"{method}_reserve_scores.npz"
        entry = ["release", "--name", method, "--report", str(method_report), "--all-scores"]
    else:
        out, report = args.out, args.report
        method = (
            args.method
            or json.loads((report / "selection.json").read_text(encoding="utf-8"))["default_method"]
        )
        pool = joblib.load(out / f"{args.roster}_{method}.joblib")
        probes, seed = len(pool["probe_ids"]), int(pool["seed"])
        names = [str(name).replace("__", "/") for name in pool["model_names"]]
        d, *_ = input_context(ROOT / "artifacts/main_v2")
        with np.load(out / "questions.npz") as data:
            rows = data["confirmation"]
        fresh, domains = d.prompt_ids[rows], d.domains[rows]
        path = out / f"{args.roster}_{method}_scores.npz"
        entry = ["confirmation", "route", "--roster", args.roster, "--scores"]
        if args.method:
            entry.extend(["--method", args.method])
        if args.pool:
            entry.extend(["--pool", str(args.pool)])
    with np.load(path) as saved:
        ids, scores = saved["prompt_ids"], saved["scores"]
    # Fix a proportional, source-stratified workload independently of the router's decisions.
    rng = np.random.default_rng(48107)
    sources, counts = np.unique(domains, return_counts=True)
    expected = args.questions * counts / counts.sum()
    sizes = np.floor(expected).astype(int)
    sizes[np.argsort(-(expected % 1))[: args.questions - sizes.sum()]] += 1
    pids = np.concatenate([rng.choice(fresh[domains == j], n, replace=False) for j, n in zip(sources, sizes)])
    rng.shuffle(pids)
    positions = {int(v): i for i, v in enumerate(ids)}
    ref = scores[:, [positions[int(v)] for v in pids]]
    prompts = read_prompts(pids, ROOT / "data/raw/question_order.csv")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/route_query.py"),
        *entry,
        "--batch-size",
        str(args.batch_size),
        "--stdin-json",
    ]
    if args.panel_device:
        cmd.extend(["--panel-device", args.panel_device])
    if args.encoder_precision:
        cmd.extend(["--encoder-precision", args.encoder_precision])
    rows, times = [], []
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="3", PYTHONIOENCODING="utf-8")
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as inbox,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log,
        sample_gpu(args.monitor_gpu) as gpu,
    ):
        for pid in pids:
            inbox.write(json.dumps({"prompt": prompts[int(pid)]}, ensure_ascii=False) + "\n")
        inbox.seek(0)
        start = perf_counter()
        with subprocess.Popen(
            cmd, stdin=inbox, stdout=subprocess.PIPE, stderr=log, env=env, text=True, encoding="utf-8"
        ) as child:
            for line in child.stdout:
                if line.startswith("{"):
                    rows.append(json.loads(line))
                    times.append(perf_counter() - start)
            if child.wait():
                log.seek(0)
                raise RuntimeError(log.read()[-3000:])
    wall = perf_counter() - start
    if len(rows) != args.questions:
        raise ValueError(f"Requested {args.questions} choices, received {len(rows)}.")
    served = np.array([[row["scores"][name] for row in rows] for name in names])
    differences = np.abs(served - ref)
    model, question = np.unravel_index(differences.argmax(), differences.shape)
    expected_names = np.asarray(names)[ref.argmax(0)]
    matched = sum(row["selected"] == peer for row, peer in zip(rows, expected_names))
    surface_only = method.startswith(("text-kernel", "text-budget-", "panel-text-", "roster-")) or method == "nominal-word"
    result = dict(
        method=method,
        roster=args.roster,
        queries=len(rows),
        seed=seed,
        probes=probes,
        encoder_precision=None if surface_only else (args.encoder_precision or "float32"),
        fitted_pool=str(args.pool) if args.pool else None,
        batch_size=args.batch_size,
        panel_device=(args.panel_device or "cpu") if published or method.startswith("panel-") or surface_only else None,
        same_choices=int(matched),
        maximum_score_gap=float(differences[model, question]),
        mean_absolute_score_gap=float(differences.mean()),
        maximum_gap_model=names[model],
        maximum_gap_prompt_id=int(pids[question]),
        first_batch_seconds=times[0],
        cold_wall_seconds=wall,
        seconds_to_last_choice=times[-1],
        seconds_per_query_after_first_batch=(times[-1] - times[args.batch_size - 1])
        / (len(rows) - args.batch_size)
        if len(rows) > args.batch_size
        else None,
        large_decoder_queries=sum(row.get("large_decoder_used", False) for row in rows)
        if "large_decoder_used" in rows[0]
        else None,
        prompt_ids=pids.tolist(),
    )
    if gpu:
        memory, power = np.asarray(gpu).T
        result.update(board_start_MiB=int(memory[0]), board_peak_MiB=int(memory.max()))
        if np.isfinite(power).all():
            result["mean_board_watts"] = float(power.mean())
    suffix = "_cuda" if args.panel_device == "cuda" else ""
    if args.encoder_precision == "bfloat16":
        suffix += "_bfloat16"
    atomic_json(args.output or report / f"serving_batch_{args.roster}_{method}{suffix}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "prompt_ids"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method")
    p.add_argument(
        "--pool", type=Path, help="Route with a newly fitted pool and compare with the stored experiment."
    )
    p.add_argument("--roster", choices=("audited_13", "expanded_37", "published_112"), default="audited_13")
    p.add_argument("--questions", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--panel-device", choices=("cpu", "cuda"))
    p.add_argument("--encoder-precision", choices=("float32", "bfloat16"))
    p.add_argument(
        "--monitor-gpu", action="store_true", help="Sample NVIDIA board memory and power every half second."
    )
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    p.add_argument("--output", type=Path, help="Save a separate workload or batch-size measurement.")
    measure(p.parse_args())
