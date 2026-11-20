"""Calibrate conditional rescue odds across historical cold-start episodes."""

import numpy as np
from scipy.special import expit, logit

from .conditional_capability import paired_profile
from .paired_distillation import behavioral_basis, behavioral_design, infer_teacher


def cohort_context(h, dom, design):
    d = int(dom.max()) + 1
    pop = np.bincount(dom)
    mass = 0.5 / len(dom) + 0.5 / (d * pop[dom])
    xtw = design.T * (mass * len(dom))
    gram = xtw @ design + 12 * np.eye(design.shape[1])
    mapping = np.linalg.solve(gram, xtw)
    return dict(
        history=h,
        domains=dom,
        design=design,
        mass=mass,
        mapping=mapping,
        absolute=mapping @ h.T,
        means=np.stack([h[:, dom == j].mean(1) for j in range(d)]),
        behavior=behavioral_basis(h, dom),
    )


def measured_pool(ctx, y, idx):
    h, dom, means = ctx["history"], ctx["domains"], ctx["means"]
    n, d = len(h), len(means)
    be = behavioral_design(h, dom, ctx["behavior"])
    teacher = infer_teacher(h, y, idx, dom, be, ctx["behavior"], 12.0)
    source = np.stack([teacher[:, dom == j].mean(1) for j in range(d)], axis=1)
    aggregate = teacher @ ctx["mass"]
    w, offsets, rates, matches = [], [], [], []
    for row in y:
        ix = idx[np.isfinite(row[idx])]
        target, obs, dd = row[ix], h[:, ix], dom[ix]
        weights, _ = paired_profile(obs, target[None], dd, np.bincount(dom), np.arange(d))
        weights = weights[0]
        residual = target - weights @ obs
        raw = np.array([target[dd == j].mean() if (dd == j).any() else target.mean() for j in range(d)])
        adjust = np.array(
            [(residual[dd == j].sum() + 6 * residual.mean()) / ((dd == j).sum() + 6) for j in range(d)]
        )
        disagreement = abs(obs - target).mean(1)
        w.append(weights)
        offsets.append(adjust)
        rates.append(raw)
        matches.append(np.sort(disagreement)[:3])
    # Distill absolute correctness on the shared event dictionary, including additional weight on the real observations.
    design = ctx["design"]
    pred = ctx["mapping"] @ teacher.T
    for i, row in enumerate(y):
        ix = idx[np.isfinite(row[idx])]
        dx = design[ix]
        residual = row[ix] - dx @ pred[:, i]
        pred[:, i] += np.linalg.solve(dx.T @ dx + 70 * np.eye(dx.shape[1]), dx.T @ residual) * 0.50
    profiles = dict(
        w=np.stack(w),
        offsets=np.stack(offsets),
        rates=np.stack(rates),
        matches=np.stack(matches),
        source=source,
        aggregate=aggregate,
        direct=pred.T,
        n_reference=n,
    )
    return profiles, teacher


def pair_profile(ctx, profiles, teacher, y, idx, anchor):
    dom = ctx["domains"]
    d = int(dom.max()) + 1
    pa = teacher[anchor]
    q = teacher * (1 - pa)
    r = pa * (1 - teacher)
    # The joint probability is identifiable on the shared panel. Correct conditional-independent
    # estimates in each domain with the empirical covariance of the teachers' errors.
    err = y[:, idx] - teacher[:, idx]
    for j in range(d):
        sampled = dom[idx] == j
        covariance = np.nansum(err[:, sampled] * err[anchor, sampled], axis=1) / (sampled.sum() + 10)
        take = dom == j
        lo = np.maximum(0, teacher[:, take] + pa[take] - 1)
        hi = np.minimum(teacher[:, take], pa[take])
        both = np.clip(teacher[:, take] * pa[take] + covariance[:, None], lo, hi)
        q[:, take], r[:, take] = teacher[:, take] - both, pa[take] - both
    pair_truth = y[:, idx] - y[anchor, idx]
    observed = np.stack(
        [
            ((pair_truth > 0)[:, dom[idx] == j].sum(1) + 1)
            / ((abs(pair_truth) == 1)[:, dom[idx] == j].sum(1) + 2)
            for j in range(d)
        ],
        axis=1,
    )
    return dict(rescue=ctx["mapping"] @ q.T, harm=ctx["mapping"] @ r.T, observed=observed, anchor=anchor)


def pair_inputs(ctx, pf, pair, design, dom, challenger):
    a, m = int(pair["anchor"]), challenger
    h = (design @ ctx["absolute"]).clip(0, 1)
    mu = ctx["means"][dom]
    wm, wa = pf["w"][[m, a]]
    fp = (0.4 * mu + 0.6 * h) @ np.stack((wm, wa)).T + pf["offsets"][[m, a]][:, dom].T
    absolute = design @ pf["direct"][[m, a]].T
    rescue = (design @ pair["rescue"][:, m]).clip(0.005, 0.95)
    harm = (design @ pair["harm"][:, m]).clip(0.005, 0.95)
    chance = (rescue / (rescue + harm)).clip(0.015, 0.985)
    other = pf["source"][[m, a]][:, dom].T
    rates = pf["rates"][[m, a]][:, dom].T
    const = np.r_[
        pf["aggregate"][[m, a]],
        pf["aggregate"][m] - pf["aggregate"][a],
        pf["matches"][m],
        pf["matches"][a],
        np.linalg.norm(wm - wa),
    ]
    data = np.column_stack(
        (
            np.broadcast_to(const, (len(dom), len(const))),
            other,
            other[:, 0] - other[:, 1],
            rates,
            rates[:, 0] - rates[:, 1],
            pair["observed"][m, dom],
            fp,
            fp[:, 0] - fp[:, 1],
            absolute,
            absolute[:, 0] - absolute[:, 1],
            rescue,
            harm,
            chance,
            rescue + harm,
            h.mean(1),
            h.std(1),
            design[:, 1 : 1 + len(ctx["means"]) + 12],
        )
    )
    return data.astype(np.float32), logit(chance), (rescue + harm).clip(0.01, 1)


def rescue_scores(ctx, pf, pair, design, dom, model, alpha, trees=0):
    values = []
    for c in range(len(pf["w"])):
        if c == pair["anchor"]:
            values.append(np.zeros(len(dom)))
            continue
        x, baseline, disagreement = pair_inputs(ctx, pf, pair, design, dom, c)
        correction = model.predict(x, prediction_type="RawFormulaVal", ntree_end=trees, thread_count=3)
        values.append((2 * expit(baseline + alpha * correction) - 1) * disagreement)
    return np.stack(values)
