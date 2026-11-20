"""Stop a shared probe panel using the repository's existing value-of-information criterion."""

import numpy as np
from scipy.special import softmax

from coldstart.conditional_capability import paired_profile
from coldstart.switch_calibration import peer_vectors
from pilot_routing_acquisition import expected_switch_value


def stages(cap):
    return sorted({min(cap, n) for n in (60, 90, 120, 150, 175, 200)})


def peer_scores(h, y, idx, dom, readout, reference):
    obs = y[:, idx]
    d = int(dom.max()) + 1
    if np.isfinite(obs).all():
        history = h[:, idx]
        weights, ipw = paired_profile(history, obs, dom[idx], np.bincount(dom), np.arange(d))
        distance = (obs @ ipw)[:, None] + (history @ ipw)[None, :] - 2 * (obs * ipw) @ history.T
        weights = 0.64 * weights + 0.36 * softmax(-distance / 0.029, axis=1)
        errors = obs - weights @ history
        means = [
            (errors[:, dom[idx] == j].sum(1) + 9 * errors.mean(1)) / (np.sum(dom[idx] == j) + 9)
            for j in range(d)
        ]
        off = np.stack(means, axis=1)
    else:
        weights, off = peer_vectors(h, y, idx, dom, 0.36)
    return weights @ readout[reference].T + off[:, dom[reference]]


def reference_rows(dom, balanced, seed):
    rng = np.random.default_rng(seed + 2297)
    allowed = np.setdiff1d(np.arange(len(dom)), balanced)
    weight = 0.5 / len(np.unique(dom)) + 0.5 * np.bincount(dom) / len(dom)
    return np.concatenate(
        [
            rng.choice(ix, min(len(ix), int(round(560 * v))), replace=False)
            for j, v in enumerate(weight)
            if len(ix := allowed[dom[allowed] == j])
        ]
    )


def expected_gain(h, y, idx, dom, readout, reference, seed):
    if len(y) == 1:
        return 0.0
    rng = np.random.default_rng(seed + 719 * len(idx))
    strat = [idx[dom[idx] == j] for j in np.unique(dom[idx])]
    posterior = np.stack(
        [
            peer_scores(
                h,
                y,
                np.concatenate([rng.choice(a, len(a), replace=True) for a in strat]),
                dom,
                readout,
                reference,
            )
            for _ in range(18)
        ]
    )
    mid = posterior.mean(0)
    winner = mid.argmax(0)
    differences = posterior - posterior[:, winner, np.arange(len(reference))][:, None, :]
    sd = differences.std(0, ddof=1)
    # The maximum over alternatives retains the existing two-model criterion when the pool has two members.
    return float(expected_switch_value(differences.mean(0), sd).max(0).mean())


def select_prefix(h, y, balanced, dom, readout, seed, cap, tolerance, live=False):
    ref = reference_rows(dom, balanced, seed)
    trace = []
    for count in stages(cap):
        idx = balanced[:count]
        if live and not np.isfinite(y[:, idx]).all():
            return idx, trace, "awaiting_responses"
        if count == cap:
            return idx, trace, "cap"
        gain = expected_gain(h, y, idx, dom, readout, ref, seed)
        trace.append(dict(questions=count, expected_additional_accuracy=gain))
        if tolerance > 0 and gain <= tolerance:
            return idx, trace, "evpi"
    raise ValueError("Empty panel")


def validate_stopping(runner, cfg, y, truth, dom, orders, seeds, primary_budget, root):
    import json
    from evaluate_capability_router import metrics

    published = json.loads(
        (root / "reports/routing/sequential/validation_selection.json").read_text(encoding="utf-8")
    )["evpi_tolerance"]
    tolerance = [float(v) for v in (published, 0.001, 0.002, 0.004, 0.008)]
    curves = []
    per_seed = []
    cap = int(primary_budget)
    head = runner.head(cfg.get("readout", cfg["name"]))
    for seed in seeds:
        balanced = orders["balanced", seed]
        _, trace, _ = select_prefix(runner.h, y, balanced, runner.dt, head["oof"], seed, cap, 0)
        per_seed.append((seed, balanced, trace))
    scores = {}
    for tol in (0.0, *tolerance):
        use = []
        counts = []
        for seed, balanced, trace in per_seed:
            n = next((r["questions"] for r in trace if tol and r["expected_additional_accuracy"] <= tol), cap)
            if (seed, n) not in scores:
                scores[seed, n] = runner.score(cfg, y, balanced[:n], seed)[0]
            use.append(scores[seed, n].argmax(0))
            counts.append(n)
        curves.append(
            dict(
                tolerance=tol,
                counts=counts,
                mean_questions=float(np.mean(counts)),
                **metrics(np.stack(use), truth, dom),
            )
        )
    ref = curves[0]
    acceptable = [
        row
        for row in curves[1:]
        if row["micro"] >= ref["micro"] - 0.0025
        and row["macro"] >= ref["macro"] - 0.0025
        and row["top_share"] <= ref["top_share"] + 0.02
    ]
    best = min(acceptable, key=lambda r: (r["mean_questions"], r["tolerance"])) if acceptable else ref
    return dict(
        configs=[dict(cfg, policy="stop-evpi", tolerance=best["tolerance"], probe_count=cap)],
        weights=[1],
        allowance=0.0,
        specialization=0.0,
        validation=best,
        original_tolerance=published,
        validation_curve=curves,
    )
