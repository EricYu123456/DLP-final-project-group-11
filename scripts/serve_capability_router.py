"""Plan probes, fit an LLM profile and route prompts with a validation-selected capability model."""

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from embed_new_model import ROOT, read_prompts, read_responses
from coldstart.capability import CapabilityBasis, KernelReadout, capability_query, fit_capabilities


def run(args, cfg, bank, parser):
    basis = CapabilityBasis(bank["axes"], len(bank["domain_names"]))
    if args.action in ("plan", "fit"):
        key = f"{cfg['policy']}_{args.seed}"
        count = cfg.get("probe_count", 200)
        ids, domains, history = [bank[f"{key}_{field}"][:count] for field in ("ids", "domains", "history")]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.action == "plan":
            prompts = read_prompts(ids, args.question_source)
            with args.output.open("w", encoding="utf-8", newline="") as f:
                rows = csv.writer(f)
                rows.writerow(["prompt_id", "benchmark", "prompt", "response"])
                for i, dom in zip(ids, domains):
                    rows.writerow([int(i), bank["domain_names"][dom], prompts[int(i)], ""])
            print(json.dumps({"probes": len(ids), "file": str(args.output)}))
        else:
            used, answers = read_responses(args.responses, ids)
            pos = {int(value): index for index, value in enumerate(ids)}
            idx = [pos[int(value)] for value in used]
            h = history[idx]
            start = perf_counter()
            vectors = fit_capabilities(basis, h, domains[idx], answers[None], cfg)
            fit_ms = (perf_counter() - start) * 1000
            with args.output.open("wb") as f:
                np.savez(
                    f,
                    vectors=vectors,
                    model_names=np.array([args.model_name]),
                    tier=args.tier,
                    stage=cfg["stage"],
                    seed=args.seed,
                )
            print(
                json.dumps(
                    {
                        "model_name": args.model_name,
                        "questions": int(np.isfinite(answers).sum()),
                        "dimensions": vectors.shape[1],
                        "fit_ms": round(fit_ms, 3),
                        "file": str(args.output),
                    }
                )
            )
        return

    vectors, names = [], []
    for path in args.embedding:
        with np.load(path, allow_pickle=False) as f:
            if str(f["stage"]) != cfg["stage"] or str(f["tier"]) != args.tier:
                raise ValueError(f"{path} uses another routing tier.")
            vectors.extend(f["vectors"])
            names.extend(f["model_names"].tolist())
    have_kernel = "kernel" in cfg["encoders"]
    topic_names = [n for n in cfg["encoders"] if n.startswith("topic/")]
    if topic_names:
        from coldstart.topic_readout import TopicReadout

        topics = {name: TopicReadout(args.weights / "topic", name.split("/")[1]) for name in topic_names}
    if have_kernel:
        with np.load(args.weights / "kernel.npz") as f:
            kernel = KernelReadout(f["anchors"], f["coefficients"], int(f["n_models"]))
        from sentence_transformers import SentenceTransformer

        text_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
            device="cpu",
        )
    frozen_names = [n for n in cfg["encoders"] if n.startswith("views_")]
    tuned = [n for n in cfg["encoders"] if n.startswith("gpu")]
    swap_backbones = bool(tuned and any(n.startswith("views_expert") for n in frozen_names))
    if frozen_names:
        from coldstart.question_views import FrozenQuestionViews
        from train_question_views import load_readouts

        folder = args.weights / frozen_names[0].split("/")[0]
        question_views = FrozenQuestionViews(folder.name.removeprefix("views_"))
        if swap_backbones:
            question_views.net.cpu()
        readouts = load_readouts(folder, [n.split("/")[1] for n in frozen_names])
    # Fine-tuned ensemble members share one resident backbone.
    networks = []
    if tuned:
        import torch
        from coldstart.capability_text import ResponseEncoder, tokenize_prompts, tokenizer_for

        model, spec = ResponseEncoder.restore(args.weights / f"{tuned[0]}.pt")
        tok = tokenizer_for(spec["size"])
        if len(tuned) > 1:
            networks = [
                torch.load(args.weights / f"{name}.pt", weights_only=True, map_location="cpu")["parameters"]
                for name in tuned
            ]

    def route(text):
        values_by_name = {}
        if have_kernel:
            x = text_model.encode([text], show_progress_bar=False)
            values_by_name["kernel"] = kernel.predict(x)
            for name in topic_names:
                values_by_name[name] = topics[name].predict(x, [text], values_by_name["kernel"][1])
        if frozen_names:
            if swap_backbones:
                model.cpu()
                torch.cuda.empty_cache()
                question_views.net.cuda()
            features = readouts(question_views.encode([text], batch=1))
            for name in frozen_names:
                values_by_name[name] = features[name.split("/")[1]]
            if swap_backbones:
                question_views.net.cpu()
                torch.cuda.empty_cache()
                model.cuda()
        if tuned:
            tokens = tokenize_prompts(tok, [text], spec["limit"])
            for i, name in enumerate(tuned):
                if networks:
                    model.load_state_dict(networks[i], strict=False)
                values_by_name[name] = model.predict(*tokens, batch=1)
        p, mem = [
            np.average(
                [values_by_name[n][j] for n in cfg["encoders"]], axis=0, weights=cfg.get("encoder_weights")
            )
            for j in range(2)
        ]
        design = capability_query(basis, bank["domain_means"], p, mem, cfg)
        values = np.asarray(vectors) @ design[0]
        winner = int(values.argmax())
        print(json.dumps({"selected": names[winner], "tier": args.tier}, ensure_ascii=False), flush=True)

    if args.stdin_json:
        from route_query import json_prompts

        for text in json_prompts(sys.stdin, parser):
            route(text)
    elif args.stdin:
        for line in sys.stdin:
            if line.strip():
                route(line.rstrip("\r\n"))
    else:
        route(args.prompt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    fit = sub.add_parser("fit")
    route = sub.add_parser("route")
    for cmd in (plan, fit, route):
        cmd.add_argument("--selection", type=Path, default=ROOT / "reports/routing/capability/selection.json")
        cmd.add_argument("--weights", type=Path, default=ROOT / "artifacts/capability_routing")
        cmd.add_argument(
            "--tier",
            help="Budget tier from selection.json, or 'economical'. Defaults to the best GPU validation selection.",
        )
    for cmd in (plan, fit):
        cmd.add_argument("--seed", type=int, choices=(11, 23, 37), default=11)
        cmd.add_argument("--output", type=Path, required=True)
    plan.add_argument("--question-source", type=Path, default=ROOT / "data/raw/question_order.csv")
    fit.add_argument("--responses", type=Path, required=True)
    fit.add_argument("--model-name", required=True)
    route.add_argument(
        "--embedding",
        type=Path,
        action="append",
        help="Candidate .npz. Defaults to the bundled 13-model roster.",
    )
    group = route.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--stdin", action="store_true")
    group.add_argument("--stdin-json", action="store_true")
    args = p.parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.tier = (
        selection["economical_on_validation"]
        if args.tier == "economical"
        else (args.tier or selection["best_hardware_on_validation"])
    )
    if args.tier not in selection["tiers"]:
        p.error("Available tiers are " + ", ".join(selection["tiers"]))
    if args.action == "route" and not args.embedding:
        args.embedding = [args.weights / f"audited_13_{args.tier}.npz"]
    with np.load(args.weights / "probe_bank.npz", allow_pickle=False) as bank:
        run(args, selection["tiers"][args.tier], bank, p)
    return 0
