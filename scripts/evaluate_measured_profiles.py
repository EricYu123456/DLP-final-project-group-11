"""Compare graded-only and richer model embeddings after one real 200-question onboarding per candidate."""

import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.special import softmax
import torch
from threadpoolctl import threadpool_limits

from coldstart.latent_answers import answer_arrays, answer_signature
from coldstart.nominal_response import (
    add_choice_signal,
    choice_effect,
    fit_choice_tilt,
    fit_new,
    outputs,
    query,
    scores,
)
from evaluate_capability_router import ROOT, benchmark_means, input_context, paired_gain
from measure_candidate_answers import CANDIDATES
from run_experiment import atomic_csv, atomic_json
from train_conditional_router import proxy_data
from train_nominal_router import load_net


def distributions(out, ids, names, evidence, raw):
    m, n, k = len(names), len(ids), raw["mask"].shape[1]
    target = np.zeros((m, n, k), np.float32)
    weight = np.zeros((m, n), np.float32)
    records = []
    for key, (name, *_) in CANDIDATES.items():
        file = out / "measured_candidates" / f"{key}.npz"
        if not file.is_file():
            continue
        with np.load(file) as f:
            if not np.array_equal(ids, f["prompt_ids"]) or str(f["model_name"]) != name:
                raise ValueError(f"Candidate {name} was measured on a different panel.")
            prob, words = f["probabilities"], f["answers"]
            record = dict(
                name=name,
                seconds=float(f["seconds"]),
                device=str(f["device"]),
                quantized_file=str(f["filename"]),
            )
        i = names.index(name)
        mc = raw["mc"] & (prob[:, :4].max(1) > 0.251)
        target[i, mc, :4] = prob[mc, :4]
        weight[i, mc] = 1.0
        for j in np.flatnonzero(~raw["mc"]):
            num, tokens = answer_signature(words[j])
            if not num and not tokens:
                continue
            hit = set()
            for e, (p, answer) in enumerate(evidence):
                if p.shape[-1] > 6 and not p[j, 0, 6]:
                    continue
                count, terms = answer_signature(answer[j])
                same = (num and num == count) or (
                    not num
                    and not count
                    and tokens
                    and len(tokens & terms) / max(1, len(tokens | terms)) >= 0.75
                )
                if same:
                    hit.add(int(raw["votes"][j, e].argmax()))
            if len(hit) == 1 or (not hit and num):
                target[i, j, next(iter(hit)) if hit else raw["mask"][j].sum() - 1] = 1.0
                weight[i, j] = 0.30
        records.append(
            dict(
                **record,
                unique_prompts=n,
                choice_observations=int(mc.sum()),
                free_matches=int(np.sum(weight[i] == 0.3)),
            )
        )
    return target, weight, records


def reliability(target, weight, known, truth, idx):
    selected = target[:, idx].argmax(2)
    out = np.zeros(len(target))
    for i in range(len(target)):
        ok = (weight[i, idx] == 1) & (known[idx] >= 0) & np.isfinite(truth[i, idx])
        if ok.sum() < 8:
            continue
        actual = selected[i, ok] == known[idx][ok]
        archived = truth[i, idx][ok]
        agree = np.mean(actual == archived)
        chance = actual.mean() * archived.mean() + (1 - actual.mean()) * (1 - archived.mean())
        out[i] = np.clip((agree - chance) / (1 - chance + 0.001), 0, 1) * len(actual) / (len(actual) + 15)
    return out


def setup(args):
    d, s, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr = f["training"]
        positions = {int(v): i for i, v in enumerate(tr)}
        idx = np.array([positions[int(v)] for v in f["balanced_11"][:200]])
    ordered = np.asarray(s.test_models)[np.argsort(d.model_ids[s.test_models])]
    ids = d.prompt_ids[tr[idx]]
    names = [str(d.model_names[i]) for i in ordered]
    return d, s, tr, idx, ids, names, ordered


def fit(args):
    begin = perf_counter()
    d, s, tr, idx, ids, names, roster = setup(args)
    root = args.out
    candidates = []
    for name in ("nominal-small", "nominal-families", "nominal-moe", "nominal-large"):
        path = args.report / f"{name}_selection.json"
        if path.exists():
            import pandas as pd

            table = pd.read_csv(args.report / f"{name}_validation.csv")
            rows = table[(table.probe_count == 200) & (table.top_share < 0.92)]
            if len(rows):
                candidates.append(rows.loc[rows.robust_objective.idxmax()].to_dict())
    cfg = max(candidates, key=lambda v: v["robust_objective"])
    src = root / cfg["name"]
    proxies = cfg["proxies"].split()
    ev = proxy_data(root, ids, proxies, "development")
    raw = answer_arrays(ev)
    choice, available, records = distributions(root, ids, names, ev, raw)
    if not records:
        raise ValueError("Acquire the actual candidate responses on the common panel first.")
    dt = d.domains[tr]
    h = d.labels[np.ix_(np.r_[s.train_models, s.val_models], tr)]
    y = np.full((len(roster), len(tr)), np.nan, dtype=np.float32)
    y[:, idx] = d.labels[np.ix_(roster, tr[idx])]
    ds = dt[idx]
    net = load_net(src / "refit.pt")
    with np.load(src / "refit.npz") as f:
        hist, means = f["history"], f["means"]
    with np.load(src / "keys.npz") as f:
        keys = f["keys"]
    refraw = answer_arrays(proxy_data(root, d.prompt_ids[tr], proxies, "development"))
    Q = query(net, refraw, dt)
    with np.load(root / "text/refit75/predictions.npz") as f:
        oof = f["historical"][idx]

    def profile(ix, strength):
        conf = reliability(choice, available, keys[idx], y[:, idx], ix)
        dist = torch.tensor(choice[:, ix].transpose(1, 2, 0).copy())
        importance = torch.tensor((available[:, ix] * conf[:, None] * strength).T, dtype=torch.float32)
        return fit_new(
            net, hist, means, Q, dt, keys, h, y, idx[ix], cfg["nearest"], cfg["prior"], dist, importance
        )

    sample = np.concatenate(
        [np.flatnonzero(dt == j)[:: max(1, np.sum(dt == j) // 115)] for j in np.unique(dt)]
    )

    def calibrate_choices(pf, ix):
        conf = reliability(choice, available, keys[idx], y[:, idx], ix)
        pf["choice_tilt"] = fit_choice_tilt(
            net,
            pf,
            tuple(v[idx[ix]] for v in Q),
            ds[ix],
            {k: v[ix] for k, v in raw.items()},
            choice[:, ix],
            available[:, ix] * conf[:, None],
        )
        effect = choice_effect(
            net, pf, tuple(v[sample] for v in Q), dt[sample], {k: v[sample] for k, v in refraw.items()}
        )
        pf["choice_center"] = np.stack([effect[:, dt[sample] == j].mean(1) for j in np.unique(dt)], axis=1)
        return pf

    rng = np.random.default_rng(1681)
    folds = np.empty(len(idx), int)
    for j in np.unique(ds):
        p = rng.permutation(np.flatnonzero(ds == j))
        folds[p] = np.arange(len(p)) % 4
    options = (0.0, 0.45, 1.3, 3.3)
    signals = np.array([0.0, 0.5, 1.0, 2.0])
    losses = np.zeros((len(options), len(idx)))
    choice_losses = np.zeros((len(options), len(signals), len(idx)))
    for k in range(4):
        a, b = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        for t, value in enumerate(options):
            pf = profile(a, value)
            nominal = outputs(net, pf["z"], tuple(v[idx[b]] for v in Q), ds[b])
            mat = scores(pf, nominal, oof[b], ds[b], means, cfg)
            # Center paired errors on credible contenders to select a supervision weight within the panel.
            focus = softmax(15 * pf["source"][:, ds[b]], axis=0)
            err = mat - y[:, idx[b]]
            losses[t, b] = (focus * (err - (focus * err).sum(0)) ** 2).sum(0)
            pf = calibrate_choices(pf, a)
            effect = choice_effect(
                net, pf, tuple(v[idx[b]] for v in Q), ds[b], {key: v[b] for key, v in raw.items()}
            )
            centered = effect - pf["choice_center"][:, ds[b]]
            for i, signal in enumerate(signals):
                resid = err + signal * centered
                choice_losses[t, i, b] = (focus * (resid - (focus * resid).sum(0)) ** 2).sum(0)
    diff = losses - losses[losses.mean(1).argmin()]
    probability = softmax(-diff.mean(1) / (diff.std(1) / np.sqrt(len(idx))).clip(0.00025))
    weight = float(probability @ np.array(options))
    objective = (probability[:, None, None] * choice_losses).sum(0)
    regret = objective - objective[objective.mean(1).argmin()]
    choice_probability = softmax(-regret.mean(1) / (regret.std(1) / np.sqrt(len(idx))).clip(0.00025))
    signal = float(choice_probability @ signals)
    reliability_by_model = reliability(choice, available, keys[idx], y[:, idx], np.arange(len(idx)))
    for row in records:
        row["paired_reliability"] = float(reliability_by_model[names.index(row["name"])])
    from serve_confirmation_router import readout_file

    shared = dict(head=joblib.load(root / "text/refit75/readout.joblib"), net=net, means=means)
    store = readout_file(root, cfg)
    store.parent.mkdir(exist_ok=True)
    joblib.dump(shared, store, compress=2)
    for key, value in (("measured", weight), ("measured-binary", 0.0), ("measured-choice", weight)):
        pf = profile(np.arange(len(idx)), value)
        if key == "measured-choice":
            pf = calibrate_choices(pf, np.arange(len(idx)))
            pf["choice_signal"] = signal
        joblib.dump(
            dict(
                configs=[cfg],
                profiles=[pf],
                weights=[1],
                model_names=np.array(names),
                model_ids=d.model_ids[roster],
                probe_ids=ids,
                seed=11,
                method=key,
            ),
            root / f"audited_13_{key}.joblib",
            compress=2,
        )
    atomic_json(
        args.report / "measured_selection.json",
        dict(
            seed=11,
            panel_questions=len(ids),
            candidates=len(roster),
            supervision_strength=weight,
            choice_signal=signal,
            choice_signal_options=signals.tolist(),
            choice_signal_weights=choice_probability.tolist(),
            weights=probability.tolist(),
            options=options,
            solver=cfg,
            measurement=records,
            fit_seconds=perf_counter() - begin,
            rule="Use independent development outcomes to choose the historical answer decoder. Select the effect of measured answer probabilities by four-fold cross-validation inside the same 200 graded prompts. Unmeasured candidates retain their binary profiles.",
        ),
    )
    print(
        "Selected rich-profile router", cfg["name"], weight, len(records), "real candidate models", flush=True
    )


def score(args):
    from confirmation_models import ConfirmationModels
    import pandas as pd

    d, s, tr, idx, ids, names, roster = setup(args)
    with np.load(args.out / "questions.npz") as f:
        manifest = dict(f)
    q = np.r_[manifest["confirmation"], manifest["previous_test"]]
    runner = ConfirmationModels(args, d, s, manifest, q, "evaluation", True)
    routes = {}
    for name in ("measured", "measured-binary", "measured-choice"):
        pool = joblib.load(args.out / f"audited_13_{name}.joblib")
        cfg = pool["configs"][0]
        pf = pool["profiles"][0]
        block = runner.component(cfg)
        dom = block["head"]["dom"]
        mat = scores(
            pf,
            outputs(block["net"], pf["z"], block["query"], dom),
            block["head"]["p"],
            dom,
            block["means"],
            cfg,
        )
        if "choice_tilt" in pf:
            raw = answer_arrays(proxy_data(args.out, d.prompt_ids[q], cfg["proxies"].split(), "evaluation"))
            mat = add_choice_signal(mat, block["net"], pf, block["query"], dom, raw)
        routes[name] = mat.argmax(0)
        np.savez_compressed(
            args.out / f"audited_13_{name}_scores.npz", scores=mat, prompt_ids=d.prompt_ids[q]
        )
    with np.load(args.out / "choices_75_audited_13.npz") as f:
        if not np.array_equal(d.prompt_ids[q], f["prompt_ids"]) or not np.array_equal(
            d.model_ids[roster], f["model_ids"]
        ):
            raise ValueError(
                "The measured comparison and the primary study must use identical prompts and model IDs."
            )
        for name in (
            "primary",
            "conditional",
            "cascade",
            "cascade-budget",
            "EmbedLLM-known",
            "EmbedLLM-noisy",
            "EmbedLLM-balanced-200",
            "previous-8D-refit",
            "probe-best-fixed",
        ):
            routes[name] = f[name][0]
    rows, pairs = [], []
    for phase, keep in (
        ("confirmation", np.arange(len(q)) < len(manifest["confirmation"])),
        ("previous-test", np.arange(len(q)) >= len(manifest["confirmation"])),
    ):
        tt = d.labels[np.ix_(roster, q[keep])]
        source = d.domains[q[keep]]
        byname = {n: tt[v[keep], np.arange(keep.sum())] for n, v in routes.items()}
        for n, v in routes.items():
            hit = byname[n]
            rows.append(
                dict(
                    split=phase,
                    method=n,
                    seed=11,
                    questions=int(keep.sum()),
                    macro=benchmark_means(hit, source).mean(),
                    micro=hit.mean(),
                    top_share=np.bincount(v[keep]).max() / len(hit),
                )
            )
            for method in ("measured", "measured-choice"):
                if n == method:
                    continue
                gain = byname[method] - hit
                pairs.append(
                    dict(
                        split=phase,
                        method=method,
                        reference=n,
                        **paired_gain(byname[method][None], hit[None], source),
                        rescues=int((gain > 0).sum()),
                        harms=int((gain < 0).sum()),
                    )
                )
    atomic_csv(args.report / "measured_results.csv", rows)
    atomic_csv(args.report / "measured_paired.csv", pairs)
    print(
        pd.DataFrame(rows)
        .query("method in ['measured','measured-binary','measured-choice']")
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("fit", "score"))
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    args = p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        (fit if args.phase == "fit" else score)(args)
