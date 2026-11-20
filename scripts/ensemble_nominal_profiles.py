"""Spend additional training compute on independent capability spaces using the same 200 measured questions."""

import argparse
from copy import deepcopy
from functools import lru_cache
from time import perf_counter

import joblib
import numpy as np
import torch
from evaluate_capability_router import (
    input_context,
    known_choices,
    metrics,
    test_rosters,
)
from panel_capabilities import panel_text
from replicate_nominal_profiles import DATA, FOLDER, OUT, REPORT, ROOT, SEEDS, package
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv, atomic_json
from serve_confirmation_router import readout_file
from threadpoolctl import threadpool_limits
from train_conditional_router import proxy_data

from coldstart.conditional_text import text_features
from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import outputs, query, scores, train_reference

METHODS = {
    "nominal-ensemble": (9, 0.0),
    "nominal-ensemble-three": (3, 0.0),
    "nominal-ensemble-cautious": (9, 0.5),
}


def select_member(raw, domain, keys, grades, held, seed, device, early_stop=False):
    take = ~held
    fit_raw = {k: v[take] for k, v in raw.items()}
    validation = {k: v[held] for k, v in raw.items()}
    test_domain = domain[held]
    targets = grades[:, held]
    best = {}
    checkpoints = []
    strong = [np.argsort(-grades[:, take & (domain == j)].mean(1))[:15] for j in np.unique(domain)]

    def record(step, current, z):
        net = deepcopy(current).cpu().eval()
        hist = z.detach().cpu().numpy().copy()
        pred = outputs(net, hist, query(net, validation, test_domain), test_domain)
        mse = []
        for j, roster in enumerate(strong):
            diff = (pred - targets)[roster][:, test_domain == j]
            mse.append(((diff - diff.mean(0)) ** 2).mean())
        loss = float(np.mean(mse))
        checkpoints.append({"steps": step, "risk": loss})
        improved = not best or loss < best["risk"]
        if improved:
            best.update(risk=loss, net=net, history=hist, steps=step)
        return improved or not early_stop

    train_reference(
        fit_raw,
        domain[take],
        keys[take],
        grades[:, take],
        3400,
        seed=seed,
        device=device,
        checkpoint=record,
    )
    return best, checkpoints


def ensemble_folder(out, full):
    parent = out / "nominal-replication"
    return parent / "full-ensemble" if full else parent


def compare_training(full):
    folder = ensemble_folder(OUT, full)
    rows = []
    for i in range(9):
        file = f"ensemble_member_{i}.joblib"
        original, stopped = (joblib.load(path / file) for path in (folder, folder / "early-stop"))
        original_state, stopped_state = (member["net"].state_dict() for member in (original, stopped))
        rows.append({
            "member": i,
            "equal_weights": all(torch.equal(t, stopped_state[k]) for k, t in original_state.items()),
            "equal_history": np.array_equal(original["history"], stopped["history"]),
            "equal_query_features": all(torch.equal(a, b) for a, b in zip(original["query"], stopped["query"])),
            "full_seconds": original["cost"]["seconds"],
            "early_seconds": stopped["cost"]["seconds"],
            "updates": stopped["cost"]["updates_run"],
        })
    title = "full_ensemble_nominal" if full else "ensemble_nominal"
    atomic_csv(REPORT / f"{title}_earlystop_comparison.csv", rows)
    summary = {
        "identical_members": sum(r["equal_weights"] and r["equal_history"] and r["equal_query_features"] for r in rows),
        "original_updates": 9 * 3400,
        "early_updates": sum(r["updates"] for r in rows),
        "original_seconds": sum(r["full_seconds"] for r in rows),
        "early_seconds": sum(r["early_seconds"] for r in rows),
    }
    atomic_json(REPORT / f"{title}_earlystop_cost.json", summary)
    print(summary, flush=True)


def train(device, full=False, development=False, early_stop=False):
    folder = ensemble_folder(OUT, full)
    if development:
        folder = folder / "development"
    if early_stop:
        folder = folder / "early-stop"
    folder.mkdir(parents=True, exist_ok=True)
    base = joblib.load(FOLDER / "fit.joblib")
    d, split, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(OUT / "questions.npz" if full else FOLDER / "study.npz") as f:
        tr = f["training" if full else "train"]
    if full:
        base["config"] = {**base["config"], "name": "nominal-large"}
    cfg = base["config"]
    raw = answer_arrays(proxy_data(OUT, d.prompt_ids[tr], cfg["proxies"].split(), "development"))
    group, dom = d.group_ids[tr], d.domains[tr]
    if full:
        prefix = "dev" if development else "refit"
        with (
            np.load(OUT / "nominal-large/keys.npz") as key,
            np.load(OUT / "nominal-large" / f"{prefix}.npz") as fit,
        ):
            if not np.array_equal(key["prompt_ids"], d.prompt_ids[tr]):
                raise ValueError("Full-history training and verified answer keys use different prompts.")
            from train_nominal_router import load_net

            original = load_net(OUT / "nominal-large" / f"{prefix}.pt")
            refs = split.train_models if development else np.r_[split.train_models, split.val_models]
            base.update(
                net=original,
                history=fit["history"],
                means=fit["means"],
                query=query(original, raw, dom),
                keys=key["keys"],
                domain=dom,
                grades=d.labels[np.ix_(refs, tr)],
            )
        joblib.dump(base, folder / "fit.joblib", compress=2)
    rng = np.random.default_rng(16431)
    fold = np.empty(len(group), int)
    for j in np.unique(dom):
        for i, ids in enumerate(np.array_split(rng.permutation(np.unique(group[dom == j])), 9)):
            fold[np.isin(group, ids)] = i
    nets, details, history = [], [], []
    for m in range(9):
        filename = folder / f"ensemble_member_{m}.joblib"
        if not filename.is_file():
            start = perf_counter()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            best, checkpoints = select_member(
                raw, dom, base["keys"], base["grades"], fold == m, 16431 + m, device, early_stop
            )
            cost = {
                "member": m,
                "seed": 16431 + m,
                "device": device,
                "training": int(np.sum(fold != m)),
                "validation": int(np.sum(fold == m)),
                "selected_steps": best["steps"],
                "updates_run": checkpoints[-1]["steps"],
                "selected_risk": best["risk"],
                "seconds": perf_counter() - start,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0,
            }
            pack = {
                "net": best["net"],
                "history": best["history"],
                "query": query(best["net"], raw, dom),
                "cost": cost,
                "checkpoints": checkpoints,
            }
            joblib.dump(pack, filename, compress=2)
        else:
            pack = joblib.load(filename)
        nets.append(pack["net"])
        details.append(pack["cost"])
        history.extend(dict(member=m, **r) for r in pack["checkpoints"])
        print("Shared member", details[-1], flush=True)
    joblib.dump(nets, folder / "ensemble_nets.joblib", compress=2)
    title = "full_ensemble_nominal" if full else "ensemble_nominal"
    if development:
        title += "_development"
    if early_stop:
        title += "_earlystop"
    atomic_csv(REPORT / f"{title}_training.csv", details)
    atomic_csv(REPORT / f"{title}_checkpoints.csv", history)


def joint_profiles(shared, members, grades, idx, device):
    from coldstart.nominal_response import calibrated_profile, profile_objective
    from coldstart.switch_calibration import peer_vectors

    cfg, domain = shared["config"], shared["domain"]
    peers, offset = peer_vectors(shared["grades"], grades, idx, domain, cfg["nearest"])
    net = members[0]["net"]
    centers = torch.as_tensor(
        np.stack([peers @ m["history"] for m in members]), dtype=torch.float32, device=device
    )
    centers[:, :, : net.domains] += torch.as_tensor(3.3 * offset, dtype=torch.float32, device=device)
    z = torch.nn.Parameter(centers.clone())
    query_columns = zip(*(m["query"] for m in members))
    q = tuple(torch.stack([v[idx] for v in column]).to(device) for column in query_columns)
    ids = torch.as_tensor(domain[idx], dtype=torch.long, device=device)
    key = torch.as_tensor(shared["keys"][idx], dtype=torch.long, device=device)
    target = torch.as_tensor(grades[:, idx].T.copy(), dtype=torch.float32, device=device)
    observed = target.isfinite()
    target = target.nan_to_num()

    def objective(embedding, center, questions):
        return profile_objective(
            net, embedding, center, questions, ids, target, observed, key, cfg["prior"]
        )

    loss_for_all = torch.vmap(objective)
    optimizer = torch.optim.Adam([z], lr=0.035)
    for _ in range(105):
        optimizer.zero_grad(set_to_none=True)
        loss_for_all(z, centers, q).sum().backward()
        optimizer.step()
    fitted = z.detach().cpu().numpy()
    source = peers @ shared["means"].T + offset
    return [
        calibrated_profile(m["net"], v, m["query"], domain, peers, source) for m, v in zip(members, fitted)
    ]


def fit_ensemble(shared, members, y, idx, names, ids, seed, method, device=None):
    count, penalty = METHODS[method.removesuffix("-full")]
    if device is None:
        profiles = []
        for member in members[:count]:
            local = {**shared, "net": member["net"], "query": member["query"], "history": member["history"]}
            pool = package(local, y, idx, names, ids, seed)
            profiles.append(pool["profiles"][0])
    else:
        profiles = joint_profiles(shared, members[:count], y, idx, device)
    weight = np.bincount(shared["domain"]) / len(shared["domain"])
    return {
        "method": method,
        "nominal_ensemble": True,
        "profiles": profiles,
        "configs": [shared["config"]],
        "model_names": np.asarray(names),
        "probe_ids": ids,
        "seed": seed,
        "risk_penalty": penalty,
        "anchor": int((profiles[0]["source"] @ weight).argmax()),
    }


def combine(matrices, anchor, penalty):
    mean = np.mean(matrices, axis=0)
    if penalty:
        relative = matrices - matrices[:, anchor : anchor + 1, :]
        mean -= penalty * relative.std(0)
    return mean


def member_scores(profiles, nets, items, domain, peer, means, config):
    return np.stack(
        [
            scores(profile, outputs(net, profile["z"], q, domain), peer, domain, means, config)
            for profile, net, q in zip(profiles, nets, items)
        ]
    )


@lru_cache(maxsize=1)
def inference_models(out, full=False, early_stop=False):
    folder = ensemble_folder(out, full)
    if early_stop:
        folder = folder / "early-stop"
    return joblib.load(folder / "ensemble_nets.joblib"), joblib.load(
        readout_file(out, {"algorithm": "nominal", "name": "nominal-large" if full else "nominal-200"})
    )


def raw_scores(args, pool, texts):
    from serve_confirmation_router import encode, encode_prompts

    cfg = pool["configs"][0]
    nets, ref = inference_models(args.out, pool["method"].endswith("-full"), pool.get("shared_early_stop", False))
    nets = nets[: len(pool["profiles"])]
    proxies = cfg["proxies"].split()
    x = encode_prompts(texts, precision=args.encoder_precision)
    x, evidence, _, _ = encode(texts, proxies, set(), False, x=x)
    head = ref["head"]
    peer, member = head["readout"].predict(x, text_features(head["pack"], x, texts))
    domain = member.argmax(1)
    inputs = answer_arrays([evidence[k] for k in proxies])
    items = [query(net, inputs, domain) for net in nets]
    matrices = member_scores(pool["profiles"], nets, items, domain, peer, ref["means"], cfg)
    return combine(matrices, pool["anchor"], pool["risk_penalty"])


def evaluate(full=False):
    folder = ensemble_folder(OUT, full)
    base = joblib.load(folder / "fit.joblib")
    members = [joblib.load(folder / f"ensemble_member_{m}.joblib") for m in range(9)]
    cfg = base["config"]
    d, split, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    names, rosters = test_rosters(d, split, prepared, family)
    with (
        np.load(FOLDER / "study.npz") as f,
        np.load(FOLDER / "original.npz") as orig,
        np.load(OUT / "embedllm_known_ids.npz") as known,
    ):
        tr, panels = f["train"], f["panels"]
        stages = {
            "combined-withheld": np.r_[f["novel"], f["active"], f["coverage"]],
            "confirmation": f["confirmation"],
        }
        native, confirmation = dict(orig), dict(known)
    if full:
        with np.load(OUT / "questions.npz") as f:
            tr = f["training"]
            stages = {"previous-test": f["previous_test"], "confirmation": f["confirmation"]}
    q = np.concatenate(list(stages.values()))
    split_point = len(next(iter(stages.values())))
    if not full:
        with np.load(FOLDER / "routes.npz") as f:
            if not np.array_equal(q, f["query"]):
                raise ValueError("The ensemble must use the single decoder's identical questions.")
            control = dict(f)
    texts = panel_text(d.prompt_ids[q], DATA / "raw")
    nets, ref = inference_models(OUT, full)
    h = ref["head"]
    peer, member = h["readout"].predict(d.embeddings[q], text_features(h["pack"], d.embeddings[q], texts))
    dom = member.argmax(1)
    raw = []
    for stage, qs in stages.items():
        raw.append(
            answer_arrays(
                proxy_data(
                    OUT,
                    d.prompt_ids[qs],
                    cfg["proxies"].split(),
                    "evaluation" if full or stage == "confirmation" else "development",
                )
            )
        )
    query_nets = [*nets, base["net"]] if full else nets
    items = [
        tuple(
            torch.cat([v, w])
            for v, w in zip(query(net, raw[0], dom[:split_point]), query(net, raw[1], dom[split_point:]))
        )
        for net in query_nets
    ]
    pos = {int(v): i for i, v in enumerate(tr)}
    rows, pairs, costs = [], [], []
    for roster, unordered, truth in rosters:
        order = unordered.argsort()
        mids, truth = unordered[order], truth[order]
        routes = {name: [] for name in METHODS}
        single = []
        for seed, panel in zip(SEEDS, panels):
            idx = np.array([pos[int(v)] for v in panel])
            y = np.full((len(mids), len(tr)), np.nan, np.float32)
            y[:, idx] = truth[:, panel]
            start = perf_counter()
            all_models = fit_ensemble(
                base,
                members,
                y,
                idx,
                [names[k] for k in mids],
                d.prompt_ids[panel],
                seed,
                "nominal-ensemble",
            )
            costs.append(
                {"seed": seed, "roster": roster, "members": 9, "fit_seconds": perf_counter() - start}
            )
            mats = member_scores(all_models["profiles"], nets, items, dom, peer, ref["means"], cfg)
            if full:
                profile = package(base, y, idx, [names[k] for k in mids], d.prompt_ids[panel], seed)[
                    "profiles"
                ][0]
                single.append(
                    scores(
                        profile,
                        outputs(base["net"], profile["z"], items[-1], dom),
                        peer,
                        dom,
                        ref["means"],
                        cfg,
                    ).argmax(0)
                )
            for method, (count, penalty) in METHODS.items():
                mat = combine(mats[:count], all_models["anchor"], penalty)
                routes[method].append(mat.argmax(0))
                if seed == SEEDS[0]:
                    deployed = method + ("-full" if full else "")
                    pool = {
                        **all_models,
                        "method": deployed,
                        "profiles": all_models["profiles"][:count],
                        "risk_penalty": penalty,
                    }
                    joblib.dump(pool, OUT / f"{roster}_{deployed}.joblib", compress=2)
                    np.savez_compressed(
                        OUT / f"{roster}_{deployed}_scores.npz", prompt_ids=d.prompt_ids[q], scores=mat
                    )
                    if method == "nominal-ensemble":
                        np.savez_compressed(
                            OUT / f"{roster}_{deployed}_embeddings.npz",
                            model_names=pool["model_names"],
                            probe_ids=pool["probe_ids"],
                            nominal=np.stack([p["z"] for p in pool["profiles"]], axis=1),
                            peer_weights=pool["profiles"][0]["w"],
                        )
        routes = {k: np.stack(v) for k, v in routes.items()}
        routes["nominal-200"] = np.stack(single) if full else control[roster]
        if full:
            routes["EmbedLLM-known"] = known_choices(confirmation, d, mids, q, SEEDS)[0]
        else:
            held = (
                native["weight"][mids] @ d.embeddings[q[:split_point]].T + native["bias"][mids, None]
            ).argmax(0)
            same = known_choices(confirmation, d, mids, q[split_point:], SEEDS)[0]
            routes["EmbedLLM-known"] = np.column_stack((np.tile(held, (len(SEEDS), 1)), same))
        for stage, qs in stages.items():
            ix = np.flatnonzero(np.isin(q, qs) & np.isfinite(truth[:, q]).all(0))
            y = truth[:, q[ix]]
            ds = d.domains[q[ix]]
            hit = {}
            for method, value in routes.items():
                v = value[:, ix]
                rows.append(
                    dict(roster=roster, split=stage, method=method, questions=len(ix), **metrics(v, y, ds))
                )
                hit[method] = y[v, np.arange(len(ix))]
            for method in METHODS:
                for reference in ("nominal-200", "EmbedLLM-known"):
                    delta = hit[method] - hit[reference]
                    pairs.append(
                        dict(
                            roster=roster,
                            split=stage,
                            method=method,
                            reference=reference,
                            gain=delta.mean(),
                            wins=int((delta.mean(1) > 0).sum()),
                            **two_way_interval(delta, ds, d.group_ids[q[ix]]),
                        )
                    )
        print("Ensemble roster", roster, flush=True)
    for key, data in (("results", rows), ("costs", costs), ("paired", pairs)):
        title = "full_ensemble_nominal" if full else "ensemble_nominal"
        atomic_csv(REPORT / f"{title}_{key}.csv", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "compare-training"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--early-stop", action="store_true", help="Finish training a member after its first held-out risk increase.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase == "train":
            train(args.device, args.full_history, early_stop=args.early_stop)
        elif args.phase == "compare-training":
            compare_training(args.full_history)
        else:
            if args.early_stop:
                parser.error("--early-stop applies to shared model training.")
            evaluate(args.full_history)
