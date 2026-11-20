"""Replicate grade-only nominal model embeddings on independent panels and held-out historical questions."""

import argparse
import json
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from coldstart.conditional_text import ConditionalReadout, text_features, text_pack
from coldstart.latent_answers import answer_arrays
from coldstart.nominal_response import fit_new, outputs, query, scores, train_reference
from cover_release_panels import DATA, disjoint_bank
from evaluate_capability_router import ROOT, input_context, metrics, test_rosters
from panel_capabilities import panel_text
from replicate_release_panels import two_way_interval
from run_experiment import atomic_csv, atomic_json
from serve_confirmation_router import readout_file
from train_conditional_router import proxy_data

OUT = ROOT / "artifacts/capability_routing/confirmation"
FOLDER = OUT / "nominal-replication"
REPORT = ROOT / "reports/routing/capability/confirmation"
SEEDS = tuple(range(9200, 9215))
EXTRA_PEERS = {
    "nominal-200": (),
    "nominal-una": ("una-gated",),
    "nominal-peers": ("una-gated", "koala-gated"),
}


def study_file(method, name):
    return FOLDER / (name if method == "nominal-200" else f"{method}_{name}")


def prepare():
    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    _, rosters = test_rosters(d, s, prepared, family)
    with np.load(OUT / "questions.npz") as f:
        tr = f["training"]
        quotas = np.bincount(d.domains[f["balanced_11"][:200]])
        old = f["confirmation"]
    upper = OUT.parent / "release_confirmation"
    with (
        np.load(DATA / "processed/embedllm.npz") as raw,
        np.load(upper / "coverage/reserve.npz") as cover,
        np.load(OUT / "active-peer-states/reserve.npz") as active,
        np.load(upper / "coverage/panels.npz") as panels,
        np.load(upper / "replication/reserve.npz") as earlier,
    ):
        g, sp = raw["group_ids"], raw["original_split"]
        held, seen = disjoint_bank(g, sp)
        seen.update(g[panels["coverage"]].ravel())
        seen.update(g[panels["control"]].ravel())
        a, b = active["groups"], cover["groups"]
        audit = tr[~np.isin(d.group_ids[tr], list(held | seen | set(b)))]
        reserve = np.unique(np.r_[a, b, d.group_ids[audit]])
        ids = np.flatnonzero((sp == 0) & np.isin(g, reserve))
        train = tr[~np.isin(d.group_ids[tr], np.union1d(reserve, earlier["groups"]))]
    stages = dict(
        novel=audit,
        active=tr[np.isin(d.group_ids[tr], a)],
        coverage=tr[np.isin(d.group_ids[tr], b)],
        confirmation=old,
    )
    if set(d.group_ids[train]) & set(
        d.group_ids[np.concatenate(list(stages.values()))]
    ):
        raise ValueError("All query groups must stay outside the shared training bank.")
    valid = train[np.isfinite(rosters[1][2][:, train]).all(0)]
    panels = []
    for seed in SEEDS:
        gen = np.random.default_rng(seed)
        shuffled = gen.permutation(valid)
        _, first = np.unique(d.group_ids[shuffled], return_index=True)
        bank = shuffled[np.sort(first)]
        panels.append(
            np.concatenate(
                [
                    gen.choice(bank[d.domains[bank] == j], n, replace=False)
                    for j, n in enumerate(quotas)
                ]
            )
        )
    FOLDER.mkdir(exist_ok=True)
    np.savez_compressed(FOLDER / "reserve.npz", groups=reserve, prompt_ids=ids)
    np.savez_compressed(
        FOLDER / "study.npz", train=train, panels=panels, seeds=SEEDS, **stages
    )
    atomic_json(
        REPORT / "nominal_replication_sample.json",
        dict(
            training=len(train),
            reference_models=len(s.train_models) + len(s.val_models),
            panel_quotas=quotas.tolist(),
            seeds=SEEDS,
            stages={key: len(v) for key, v in stages.items()},
            novel_sources=np.bincount(d.domains[audit], minlength=5).tolist(),
            refit_withheld_groups=len(reserve),
        ),
    )


def package(shared, y, idx, model_names, ids, seed):
    cfg = shared["config"]
    p = fit_new(
        shared["net"],
        shared["history"],
        shared["means"],
        shared["query"],
        shared["domain"],
        shared["keys"],
        shared["grades"],
        y,
        idx,
        cfg["nearest"],
        cfg["prior"],
    )
    return dict(
        configs=[cfg],
        profiles=[p],
        weights=[1],
        model_names=np.asarray(model_names),
        probe_ids=ids,
        seed=seed,
        method=cfg["name"],
    )


def fit(method="nominal-200"):
    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    names, rosters = test_rosters(d, s, prepared, family)
    with np.load(FOLDER / "study.npz") as f:
        tr, panels = f["train"], f["panels"]
        stages = {k: f[k] for k in ("novel", "active", "coverage", "confirmation")}
    cfg = json.loads((REPORT / "measured_selection.json").read_text(encoding="utf-8"))[
        "solver"
    ]
    proxies = [*cfg["proxies"].split(), *EXTRA_PEERS[method]]
    cfg = dict(cfg, name=method, proxies=" ".join(proxies))
    ds = d.domains[tr]
    ref = np.r_[s.train_models, s.val_models]
    h = d.labels[np.ix_(ref, tr)]
    evidence = proxy_data(OUT, d.prompt_ids[tr], proxies, "development")
    raw = answer_arrays(evidence)
    if EXTRA_PEERS[method]:
        from training_answer_targets import resolve

        keys = resolve(OUT, d.prompt_ids[tr], evidence, raw)
    else:
        with np.load(OUT / "nominal-large/keys.npz") as f:
            index = {int(v): i for i, v in enumerate(f["prompt_ids"])}
            keys = f["keys"][[index[int(v)] for v in d.prompt_ids[tr]]]
    start = perf_counter()
    net, hist = train_reference(raw, ds, keys, h, 850)
    means = np.stack([h[:, ds == j].mean(1) for j in np.unique(ds)])
    if EXTRA_PEERS[method]:
        frozen = joblib.load(readout_file(OUT, dict(cfg, name="nominal-200")))["head"]
        basis, head = frozen["pack"], frozen["readout"]
    else:
        texts = panel_text(d.prompt_ids[tr], DATA / "raw")
        basis = text_pack(d.embeddings[tr], texts)
        features = text_features(basis, d.embeddings[tr], texts)
        head = ConditionalReadout.fit(d.embeddings[tr], features, h, ds)
    service = dict(
        net=net, means=means, head=dict(pack=basis, readout=head, proxies=[])
    )
    path = readout_file(OUT, cfg)
    path.parent.mkdir(exist_ok=True)
    joblib.dump(service, path, compress=2)
    shared = dict(
        net=net,
        history=hist,
        means=means,
        query=query(net, raw, ds),
        keys=keys,
        domain=ds,
        grades=h,
        config=cfg,
    )
    joblib.dump(shared, study_file(method, "fit.joblib"), compress=2)
    history_seconds = perf_counter() - start
    segments = []
    for name, q in stages.items():
        t = panel_text(d.prompt_ids[q], DATA / "raw")
        p, m = head.predict(d.embeddings[q], text_features(basis, d.embeddings[q], t))
        source = m.argmax(1)
        data = answer_arrays(
            proxy_data(
                OUT,
                d.prompt_ids[q],
                proxies,
                "evaluation" if name == "confirmation" else "development",
            )
        )
        segments.append((q, p, source, query(net, data, source)))
    all_q = np.concatenate(list(stages.values()))
    positions = {int(v): i for i, v in enumerate(tr)}
    costs, routes = [], {}
    for roster, unordered, truth in rosters:
        order = unordered.argsort()
        mids = unordered[order]
        by_seed, fixed = [], []
        for seed, panel in zip(SEEDS, panels):
            idx = np.array([positions[int(v)] for v in panel])
            y = np.full((len(order), len(tr)), np.nan, np.float32)
            y[:, idx] = truth[np.ix_(order, panel)]
            start = perf_counter()
            pool = package(
                shared, y, idx, [names[i] for i in mids], d.prompt_ids[panel], seed
            )
            seconds = perf_counter() - start
            profile = pool["profiles"][0]
            mat = np.column_stack(
                [
                    scores(
                        profile,
                        outputs(net, profile["z"], q, source),
                        p,
                        source,
                        means,
                        cfg,
                    )
                    for _, p, source, q in segments
                ]
            )
            chosen = mat.argmax(0)
            by_seed.append(chosen)
            fixed.append(
                np.full(
                    len(chosen),
                    (profile["source"] @ (np.bincount(ds) / len(ds))).argmax(),
                )
            )
            costs.append(
                dict(
                    roster=roster,
                    seed=seed,
                    probes=len(panel),
                    fit_seconds=seconds,
                    shared_seconds=history_seconds,
                )
            )
            if seed == SEEDS[0]:
                pool["model_ids"] = mids
                joblib.dump(pool, OUT / f"{roster}_{method}.joblib", compress=2)
                np.savez_compressed(
                    OUT / f"{roster}_{method}_scores.npz",
                    prompt_ids=d.prompt_ids[all_q],
                    scores=mat,
                )
                np.savez_compressed(
                    OUT / f"{roster}_{method}_embeddings.npz",
                    model_ids=mids,
                    model_names=pool["model_names"],
                    nominal=profile["z"],
                    peer_weights=profile["w"],
                    source=profile["source"],
                    expected=profile["expected"],
                    probe_ids=d.prompt_ids[panel],
                )
        routes[roster] = by_seed
        routes[roster + "_fixed"] = fixed
        print("Fitted replicated nominal profiles", roster, history_seconds, flush=True)
    np.savez_compressed(study_file(method, "routes.npz"), query=all_q, **routes)
    title = "nominal_replication" if method == "nominal-200" else method
    atomic_csv(REPORT / f"{title}_costs.csv", costs)


def score(target="nominal-200"):
    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    _, rosters = test_rosters(d, s, prepared, family)
    rows, pairs = [], []
    with (
        np.load(FOLDER / "study.npz") as study,
        np.load(study_file(target, "routes.npz")) as saved,
        np.load(FOLDER / "original.npz") as native,
    ):
        q = saved["query"]
        peers = {}
        for reference in list(EXTRA_PEERS)[: list(EXTRA_PEERS).index(target)]:
            with np.load(study_file(reference, "routes.npz")) as r:
                if not np.array_equal(q, r["query"]):
                    raise ValueError("Every peer ablation must use the same questions.")
                peers[reference] = {roster: r[roster] for roster, _, _ in rosters}
        stages = {
            k: np.flatnonzero(np.isin(q, study[k]))
            for k in ("novel", "active", "coverage", "confirmation")
        }
        stages["combined-withheld"] = np.concatenate(
            [stages[k] for k in ("novel", "active", "coverage")]
        )
        for roster, unordered, truth in rosters:
            order = unordered.argsort()
            ids = unordered[order]
            with np.load(OUT / f"choices_75_{roster}.npz") as f:
                old = f["EmbedLLM-known"][0, : len(stages["confirmation"])]
            with np.load(OUT / "embedllm_known_ids.npz") as f:
                original = dict(f)
            for name, cols in stages.items():
                ix = cols[np.isfinite(truth[np.ix_(order, q[cols])]).all(0)]
                y = truth[np.ix_(order, q[ix])]
                w, b = native["weight"][ids], native["bias"][ids]
                if name == "confirmation":
                    from evaluate_capability_router import known_choices

                    det, noisy = known_choices(original, d, ids, q[ix], SEEDS)
                    if not np.array_equal(det[0], old[np.isin(cols, ix)]):
                        raise ValueError(
                            "The original deterministic control must reproduce the confirmation routing."
                        )
                else:
                    z = w @ d.embeddings[q[ix]].T + b[:, None]
                    det = np.tile(z.argmax(0), (len(SEEDS), 1))
                    sd = 0.05 * np.linalg.norm(w, axis=1, keepdims=True)
                    noisy = np.stack(
                        [
                            (
                                z + sd * np.random.default_rng(v).normal(size=z.shape)
                            ).argmax(0)
                            for v in SEEDS
                        ]
                    )
                choices = {
                    target: saved[roster][:, ix],
                    "panel-fixed": saved[roster + "_fixed"][:, ix],
                    "EmbedLLM-known": det,
                    "EmbedLLM-noisy": noisy,
                }
                choices.update({key: v[roster][:, ix] for key, v in peers.items()})
                domain = d.domains[q[ix]]
                hit = {k: y[v, np.arange(len(ix))] for k, v in choices.items()}
                for method, v in choices.items():
                    rows.append(
                        dict(
                            roster=roster,
                            split=name,
                            method=method,
                            questions=len(ix),
                            **metrics(v, y, domain),
                        )
                    )
                    if method == target:
                        continue
                    gap = hit[target] - hit[method]
                    pairs.append(
                        dict(
                            roster=roster,
                            split=name,
                            reference=method,
                            micro=hit[target].mean(),
                            gain=gap.mean(),
                            wins=int((gap.mean(1) > 0).sum()),
                            rescues=(gap > 0).sum(1).mean(),
                            harms=(gap < 0).sum(1).mean(),
                            **two_way_interval(gap, domain, d.group_ids[q[ix]]),
                        )
                    )
    title = "nominal_replication" if target == "nominal-200" else target
    atomic_csv(REPORT / f"{title}_results.csv", rows)
    atomic_csv(REPORT / f"{title}_paired.csv", pairs)
    print(
        pd.DataFrame(rows).query("method==@target").to_string(index=False),
        flush=True,
    )


def full_history():
    from types import SimpleNamespace
    from confirmation_models import ConfirmationModels
    from evaluate_capability_router import known_choices
    from train_nominal_router import load_net

    d, s, _, _, prepared, family = input_context(ROOT / "artifacts/main_v2")
    names, rosters = test_rosters(d, s, prepared, family)
    with np.load(OUT / "questions.npz") as f:
        split = dict(f)
    with np.load(FOLDER / "study.npz") as f:
        panels = f["panels"]
    cfg = json.loads((REPORT / "measured_selection.json").read_text(encoding="utf-8"))[
        "solver"
    ]
    tr = split["training"]
    cut = len(split["confirmation"])
    q = np.r_[split["confirmation"], split["previous_test"]]
    runner = ConfirmationModels(
        SimpleNamespace(out=OUT, run=ROOT / "artifacts/main_v2", report=REPORT),
        d,
        s,
        split,
        q,
        "evaluation",
        True,
    )
    block = runner.component(cfg)
    original = OUT / cfg["name"]
    net = load_net(original / "refit.pt")
    with (
        np.load(original / "refit.npz") as f,
        np.load(original / "keys.npz") as k,
        np.load(OUT / "embedllm_known_ids.npz") as baseline,
    ):
        h, means, keys, native = f["history"], f["means"], k["keys"], dict(baseline)
        if not np.array_equal(k["prompt_ids"], d.prompt_ids[tr]):
            raise ValueError(
                "Original training-set answer keys do not match the nominal decoder."
            )
    raw = answer_arrays(
        proxy_data(OUT, d.prompt_ids[tr], cfg["proxies"].split(), "development")
    )
    shared = dict(
        net=net,
        history=h,
        means=means,
        query=query(net, raw, d.domains[tr]),
        domain=d.domains[tr],
        keys=keys,
        grades=d.labels[np.ix_(np.r_[s.train_models, s.val_models], tr)],
        config=cfg,
    )
    pos = {int(v): i for i, v in enumerate(tr)}
    rows, pairs = [], []
    ds = block["head"]["dom"]
    for roster, unordered, truth in rosters:
        order = unordered.argsort()
        ids = unordered[order]
        routes, anchor = [], []
        # Seed 11 checks the unchanged published binary estimator separately from the fifteen replication panels.
        for seed, panel in [(11, split["balanced_11"][:200]), *zip(SEEDS, panels)]:
            idx = np.array([pos[int(v)] for v in panel])
            y = np.full((len(ids), len(tr)), np.nan, np.float32)
            y[:, idx] = truth[np.ix_(order, panel)]
            pack = package(
                shared, y, idx, [names[i] for i in ids], d.prompt_ids[panel], seed
            )
            pf = pack["profiles"][0]
            mat = scores(
                pf,
                outputs(net, pf["z"], block["query"], ds),
                block["head"]["p"],
                ds,
                means,
                cfg,
            )
            if seed == 11:
                if roster == "audited_13":
                    with np.load(
                        OUT / "audited_13_measured-binary_scores.npz"
                    ) as check:
                        if not np.array_equal(mat.argmax(0), check["scores"].argmax(0)):
                            raise ValueError(
                                "The unchanged seed-11 binary estimator must reproduce the original choices."
                            )
                continue
            routes.append(mat.argmax(0))
            anchor.append(
                np.full(
                    len(q),
                    (pf["source"] @ (np.bincount(d.domains[tr]) / len(tr))).argmax(),
                )
            )
        for name, cols in (
            ("confirmation", np.arange(cut)),
            ("previous-test", np.arange(cut, len(q))),
        ):
            ix = cols[np.isfinite(truth[np.ix_(order, q[cols])]).all(0)]
            tt = truth[np.ix_(order, q[ix])]
            det, noisy = known_choices(native, d, ids, q[ix], SEEDS)
            policies = {
                "nominal-200": np.asarray(routes)[:, ix],
                "panel-fixed": np.asarray(anchor)[:, ix],
                "EmbedLLM-known": det,
                "EmbedLLM-noisy": noisy,
            }
            target = tt[policies["nominal-200"], np.arange(len(ix))]
            domain = d.domains[q[ix]]
            for method, v in policies.items():
                rows.append(
                    dict(
                        roster=roster,
                        split=name,
                        method=method,
                        questions=len(ix),
                        **metrics(v, tt, domain),
                    )
                )
                if method != "nominal-200":
                    gap = target - tt[v, np.arange(len(ix))]
                    pairs.append(
                        dict(
                            roster=roster,
                            split=name,
                            reference=method,
                            accuracy=target.mean(),
                            gain=gap.mean(),
                            wins=(gap.mean(1) > 0).sum(),
                            rescues=(gap > 0).sum(1).mean(),
                            harms=(gap < 0).sum(1).mean(),
                            **two_way_interval(gap, domain, d.group_ids[q[ix]]),
                        )
                    )
    atomic_csv(REPORT / "nominal_full_history_results.csv", rows)
    atomic_csv(REPORT / "nominal_full_history_paired.csv", pairs)
    print(
        pd.DataFrame(rows).query("method=='nominal-200'").to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=("prepare", "fit", "score", "full-history"), required=True
    )
    p.add_argument("--method", choices=EXTRA_PEERS, default="nominal-200")
    args = p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.phase in ("fit", "score"):
            {"fit": fit, "score": score}[args.phase](args.method)
        else:
            {"prepare": prepare, "full-history": full_history}[args.phase]()
