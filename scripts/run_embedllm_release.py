"""Compare the original 112-ID release with a router trained on 200 shared graded prompts."""

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ROOT, benchmark_means, paired_gain, read_json
from run_experiment import atomic_csv, atomic_json
from panel_capabilities import RELEASE_SEEDS as SEEDS, pack_panel, panel_text, public_routes, select_panel


def chunks(source):
    return pd.read_csv(
        source,
        usecols=["prompt_id", "model_id", "label"],
        dtype={"prompt_id": "int32", "model_id": "int16", "label": "int8"},
        chunksize=250_000,
    )


def panel(vectors, domains, bank, seed):
    rng = np.random.default_rng(seed)
    groups = [bank[domains[bank] == j] for j in np.unique(domains[bank])]
    sizes = np.sqrt([len(g) for g in groups])
    extra = 80 * sizes / sizes.sum()
    counts = 12 + np.floor(extra).astype(int)
    counts[np.argsort(-(extra % 1))[: 200 - counts.sum()]] += 1
    selected = []
    for group, count in zip(groups, counts):
        v = vectors[group]
        direction = (v - v.mean(0)) @ rng.normal(size=v.shape[1])
        strata = np.array_split(group[np.argsort(direction)], count)
        selected.extend(int(rng.choice(g)) for g in strata)
    return np.asarray(selected)


def stratified_panel(x, dom, bank, meta, texts, seed):
    """Allocate by benchmark prevalence, then cover subjects and prompt formats at equal-mass intervals."""
    from panel_rasch import prompt_subtype

    rng = np.random.default_rng(seed)
    sizes = np.bincount(dom[bank])
    extra = (200 - 6 * len(sizes)) * sizes / sizes.sum()
    counts = 6 + np.floor(extra).astype(int)
    counts[np.argsort(-(extra % 1))[: 200 - counts.sum()]] += 1
    excluded_groups = meta.loc[meta.original_split != "train", "group_id"].unique()
    clean = ~meta.group_id.isin(excluded_groups).to_numpy()
    selected = []
    for j, count in enumerate(counts):
        local = bank[(dom[bank] == j) & clean[bank]]
        group_ids = meta.group_id.to_numpy()[local]
        category = meta.categories.to_numpy()[local]
        kind = [prompt_subtype(t) for t in texts[local]]
        direction = x[local] @ rng.standard_normal(x.shape[1])
        order = np.lexsort((direction, category, kind))
        available = np.ones(len(order), dtype=bool)
        for stratum in np.array_split(np.arange(len(order)), count):
            positions = stratum[available[stratum]]
            if not len(positions):
                remaining = np.flatnonzero(available)
                positions = remaining[np.argsort(abs(remaining - np.mean(stratum)))[: max(1, len(stratum))]]
            pick = order[rng.choice(positions)]
            selected.append(local[pick])
            available[group_ids[order] == group_ids[pick]] = False
    return np.asarray(selected)


def prepare(args):
    with np.load(args.prepared) as f:
        x, domains, split, ids = (
            f[k] for k in ("embeddings", "benchmark_ids", "original_split", "prompt_ids")
        )
    bank = np.flatnonzero(split == 0)
    if args.panel_design == "stratified":
        meta = pd.read_csv(args.prepared.with_name("prompts.csv")).set_index("prompt_id").loc[ids]
        texts = panel_text(ids, args.raw)
        panels = {seed: stratified_panel(x, domains, bank, meta, texts, seed) for seed in SEEDS}
    else:
        panels = {seed: panel(x, domains, bank, seed) for seed in SEEDS}
    needed = np.unique(np.concatenate([ids[sel] for sel in panels.values()]))
    lookup = np.full(ids.max() + 1, -1, dtype=int)
    lookup[needed] = np.arange(len(needed))
    selected = np.full((112, len(needed)), np.nan, dtype=np.float32)
    for part in chunks(args.raw / "train.csv"):
        cols = lookup[part.prompt_id.to_numpy()]
        keep = cols >= 0
        # Match evaluator_router's dictionary overwrite for repeated source records.
        short = part.loc[keep].drop_duplicates(["model_id", "prompt_id"], keep="last")
        selected[short.model_id, lookup[short.prompt_id]] = short.label
    if not np.isfinite(selected).all():
        raise ValueError("Incomplete 200-prompt panel in the published training CSV.")
    for seed, pos in panels.items():
        np.savez(
            args.out / f"panel_{seed}.npz",
            x=x[pos],
            domains=domains[pos],
            prompt_ids=ids[pos],
            y=selected[:, lookup[ids[pos]]],
        )
    print(
        "Saved three independent, stratified 200-prompt panels from the published training split.", flush=True
    )


def original(args):
    vals = pd.concat(chunks(args.raw / "train.csv"), ignore_index=True)
    with np.load(args.prepared) as f:
        x, prompt_ids = f["embeddings"], f["prompt_ids"]
        test_ids = prompt_ids[f["original_split"] == 2]
        if getattr(args, "reserve", None) is not None:
            with np.load(args.reserve) as reserve:
                excluded = np.isin(f["group_ids"], reserve["groups"])
                test_ids = reserve["prompt_ids"]
            vals = vals.loc[~excluded[vals.prompt_id.to_numpy()]].reset_index(drop=True)
    if not np.array_equal(prompt_ids, np.arange(len(prompt_ids))):
        raise ValueError("Expected the original prompt-indexed feature bank.")
    models = torch.as_tensor(vals.model_id.to_numpy(dtype=np.int64), device="cuda")
    qs = torch.as_tensor(vals.prompt_id.to_numpy(dtype=np.int64), device="cuda")
    truth = torch.as_tensor(vals.label.to_numpy(dtype=np.int64), device="cuda")
    torch.manual_seed(42)
    source = args.official_mf or args.out / "upstream_mf.py"
    spec = importlib.util.spec_from_file_location("embedllm_public_mf", source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    net = upstream.TextMF(torch.from_numpy(x), 232, 0.05, 112, len(x)).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    start = perf_counter()
    for epoch in range(50):
        for first in range(0, len(qs), 2048):
            ix = slice(first, first + 2048)
            logits = net(models[ix], qs[ix])
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(logits, truth[ix]).backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            torch.cuda.synchronize()
            print(f"Published TextMF epoch {epoch + 1}/50 after {perf_counter() - start:.0f}s", flush=True)
    with torch.no_grad():
        torch.cuda.synchronize()
        training_seconds = perf_counter() - start
        routing_start = perf_counter()
        # The authors' router forwards all 112 candidates for one query, then takes the logit difference.
        net.eval()
        candidates = torch.arange(112, device="cuda")
        test_questions = torch.as_tensor(test_ids, device="cuda")
        native_routes = []
        for seed in (None, *SEEDS):
            if seed is not None:
                torch.cuda.manual_seed(seed)
            choices = []
            for item in test_questions:
                logits = net(candidates, item.expand_as(candidates), test_mode=(seed is None))
                choices.append((logits[:, 1] - logits[:, 0]).argmax())
            native_routes.append(torch.stack(choices).cpu().numpy())
        diff = net.classifier.weight[1] - net.classifier.weight[0]
        w = net.P.weight @ (diff[:, None] * net.text_proj.weight)
        b = net.P.weight @ (diff * net.text_proj.bias)
        np.savez(
            args.out / "original.npz",
            weight=w.cpu().numpy(),
            bias=b.cpu().numpy(),
            rows=len(qs),
            seconds=training_seconds,
            routing_seconds=perf_counter() - routing_start,
            prompt_ids=test_ids,
            deterministic=native_routes[0],
            noisy=np.stack(native_routes[1:]),
            used_upstream_class=True,
        )


def extract_views(args, test):
    from coldstart.question_views import FrozenQuestionViews
    from train_capability_router import prompt_texts, VramMeter

    if test:
        selected = read_json(args.out / "decision.json")
        if all(row["encoder"] != "views" or not row["signal"] for row in selected.values()):
            print("All panels selected MPNet.", flush=True)
            return
        with np.load(args.prepared) as f:
            ids = f["prompt_ids"][f["original_split"] == 2]
    else:
        panels = []
        for seed in SEEDS:
            with np.load(args.out / f"panel_{seed}.npz") as f:
                panels.append(f["prompt_ids"])
        ids = np.unique(np.concatenate(panels))
    start = perf_counter()
    meter = VramMeter(16)
    try:
        encoder = FrozenQuestionViews("expert")
        texts = prompt_texts(args.raw / "question_order.csv", ids)
        features = np.vstack([encoder.encode(texts[i : i + 512], 32) for i in range(0, len(texts), 512)])
        phase = "test" if test else "support"
        np.savez(
            args.out / f"views_{phase}.npz",
            ids=ids,
            features=features,
            seconds=perf_counter() - start,
            board_peak_mib=meter.peak,
        )
        print(f"Encoded {len(ids)} {phase} questions in {perf_counter() - start:.1f}s", flush=True)
    finally:
        meter.close()


def export_release_panel(args, seed, cfg):
    with np.load(args.out / f"panel_{seed}.npz") as panel:
        x, domains, y, ids = (panel[k] for k in ("x", "domains", "y", "prompt_ids"))
    raw = None
    if cfg["encoder"] == "views":
        with np.load(args.out / "views_support.npz") as source:
            raw = source["features"][np.searchsorted(source["ids"], ids)]
    names = pd.read_csv(args.raw / "model_order.csv").sort_values("model_id")["model_name"].to_numpy(str)
    if cfg.get("method") == "rasch":
        from panel_rasch import pack_rasch

        packed = pack_rasch(x, domains, y, panel_text(ids, args.raw), ids, names, cfg)
    else:
        packed = pack_panel(x, domains, y, panel_text(ids, args.raw), ids, names, cfg, raw)
    np.savez_compressed(args.out / f"router_{seed}.npz", **packed)


def route_prompt(args):
    with np.load(args.router or args.out / f"router_{args.seed}.npz") as router:
        x, views = None, None
        if float(router["signal"]):
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2",
                revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
                device="cpu",
            )
            x = model.encode([args.prompt], show_progress_bar=False)
        if str(router["encoder"]) == "views" and float(router["signal"]):
            from coldstart.question_views import FrozenQuestionViews

            views = FrozenQuestionViews("expert").encode([args.prompt], batch=1)
        best = int(public_routes(router, x, [args.prompt], views)[0])
        mid = int(router["model_ids"][best]) if "model_ids" in router else best
        print(json.dumps(dict(model_id=mid, selected=str(router["model_names"][best]))))


def freeze_release(args):
    decisions = {}
    costs = {}
    source_views = args.out / "views_support.npz"
    use_views = source_views.exists() and args.panel_design == "standard"
    if use_views:
        with np.load(source_views) as f:
            view_ids, view_bank = f["ids"], f["features"]
            costs["shared_support_text_features"] = dict(
                seconds=float(f["seconds"]), questions=len(view_ids), board_peak_mib=int(f["board_peak_mib"])
            )
    for seed in SEEDS:
        start = perf_counter()
        with np.load(args.out / f"panel_{seed}.npz") as f:
            x, domains, y, ids = f["x"], f["domains"], f["y"], f["prompt_ids"]
        texts = panel_text(ids, args.raw)
        views = view_bank[np.searchsorted(view_ids, ids)] if use_views else None
        decisions[str(seed)] = select_panel(x, domains, y, texts, seed, views, retentions=(0.5, 0.75, 1.0))
        if args.panel_design == "stratified":
            from panel_rasch import select_measurement

            names = pd.read_csv(args.raw / "model_order.csv").sort_values("model_id").model_name.to_numpy(str)
            decisions[str(seed)] = select_measurement(
                x, domains, y, texts, ids, names, seed, decisions[str(seed)]
            )
        costs[str(seed)] = dict(selection_seconds=perf_counter() - start)
    atomic_json(args.out / "decision.json", decisions)
    atomic_json(args.report / "release112_selection.json", decisions)
    for seed in SEEDS:
        start = perf_counter()
        export_release_panel(args, seed, decisions[str(seed)])
        costs[str(seed)]["fit_and_export_seconds"] = perf_counter() - start
    atomic_json(args.report / "release112_cost.json", costs)
    print(decisions, flush=True)


def score_release(args):
    with np.load(args.prepared) as f:
        idx = np.flatnonzero(f["original_split"] == 2)
        x, dom, ids = f["embeddings"][idx], f["benchmark_ids"][idx], f["prompt_ids"][idx]
    truth = np.zeros((112, len(ids)), np.float32)
    for part in chunks(args.raw / "test.csv"):
        last = part.drop_duplicates(["model_id", "prompt_id"], keep="last")
        truth[last.model_id, np.searchsorted(ids, last.prompt_id)] = last.label
    with np.load(args.reference or args.out / "original.npz") as f:
        if not np.array_equal(f["prompt_ids"], ids):
            raise ValueError("Published reference and evaluation question IDs differ.")
        deterministic, noisy = f["deterministic"], f["noisy"]
    rows = []
    outcomes = {}
    routes = {}
    query_text = panel_text(ids, args.raw)
    for run, seed in enumerate(SEEDS):
        with np.load(args.out / f"router_{seed}.npz") as router:
            views = None
            if str(router["encoder"]) == "views" and float(router["signal"]):
                with np.load(args.out / "views_test.npz") as feat:
                    if not np.array_equal(feat["ids"], ids):
                        raise ValueError("Original release query IDs differ from the feature cache.")
                    views = feat["features"]
            learned = public_routes(router, x, query_text, views)
        for name, choice in [
            ("200-only", learned),
            ("original-deterministic", deterministic),
            ("original-noisy", noisy[run]),
        ]:
            correct = truth[choice, np.arange(len(ids))]
            counts = np.bincount(choice, minlength=112)
            rows.append(
                dict(
                    method=name,
                    seed=seed,
                    queries=len(ids),
                    candidates=len(counts),
                    macro=benchmark_means(correct, dom).mean(),
                    micro=correct.mean(),
                    top_id=counts.argmax(),
                    top_share=counts.max() / len(ids),
                    models_used=int((counts > 0).sum()),
                    best_fixed_micro=float(truth.mean(1).max()),
                    same_usage_random_micro=float(counts @ truth.mean(1) / len(ids)),
                )
            )
            outcomes.setdefault(name, []).append(correct)
            routes.setdefault(name, []).append(choice)
    atomic_csv(args.report / "release112_by_seed.csv", rows)
    paired = []
    for name in ("original-deterministic", "original-noisy"):
        left, right = np.stack(outcomes["200-only"]), np.stack(outcomes[name])
        difference = left - right
        paired.append(
            dict(
                reference=name,
                **paired_gain(left, right, dom),
                mean_rescues=(difference > 0).sum(1).mean(),
                mean_harms=(difference < 0).sum(1).mean(),
                mean_switches=(np.stack(routes["200-only"]) != np.stack(routes[name])).sum(1).mean(),
            )
        )
    atomic_csv(args.report / "release112_paired.csv", paired)
    np.savez_compressed(
        args.out / "choices.npz", prompt_ids=ids, **{k: np.stack(v) for k, v in routes.items()}
    )
    print(pd.DataFrame(rows).groupby("method")[["macro", "micro", "top_share"]].mean(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        required=True,
        choices=("prepare", "original", "view-support", "freeze", "view-test", "test", "route"),
    )
    p.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    p.add_argument("--prepared", type=Path, default=ROOT / "data/processed/embedllm.npz")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/release112")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability")
    p.add_argument(
        "--official-mf", type=Path, help="Path to the authors’ algorithm/mf.py for the original phase."
    )
    p.add_argument("--panel-design", choices=("standard", "stratified"), default="standard")
    p.add_argument(
        "--reference", type=Path, help="Reuse an already trained native reference on the same release."
    )
    p.add_argument(
        "--reserve",
        type=Path,
        help="Refit the original architecture excluding these question groups and score this reserve.",
    )
    p.add_argument("--seed", type=int, choices=SEEDS, default=11)
    p.add_argument("--prompt", help="Incoming question for the route phase.")
    p.add_argument("--router", type=Path, help="Panel-only router .npz for the route phase.")
    args = p.parse_args()
    if args.phase == "route" and not args.prompt:
        p.error("The route phase requires --prompt.")
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=3):
        {
            "prepare": prepare,
            "original": original,
            "freeze": freeze_release,
            "test": score_release,
            "route": route_prompt,
            "view-support": lambda a: extract_views(a, False),
            "view-test": lambda a: extract_views(a, True),
        }[args.phase](args)
