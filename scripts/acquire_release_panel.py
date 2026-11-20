"""Select at most 200 shared questions using estimated uncertainty among the current leading models."""

import numpy as np
from scipy.special import ndtr, softmax
from sklearn.cluster import KMeans

from panel_posterior import posterior_abilities, question_map
from run_embedllm_release import chunks


def quadrature_panel(args, seed, ids, x, dom, text, aux, meta, size, source_mass, n_items):
    """Minimize the posterior variance of each source mean before observing any target responses."""
    from scipy.spatial.distance import cdist
    from population_panel import population_coordinates

    rng = np.random.default_rng(seed + 8647)
    mass = 0.65 * source_mass + 0.35 / len(source_mass)
    sources = []
    for j in np.unique(dom):
        group = np.flatnonzero(dom == j)
        xx = x[group].astype(np.float64)
        xx /= np.linalg.norm(xx, axis=1, keepdims=True).clip(1e-6)
        coords, _ = population_coordinates(xx, xx, aux[group], aux[group], text[group], text[group])
        dist = cdist(coords, coords, metric="sqeuclidean") / max(1, coords.shape[1])
        kernel = 0.48 * np.exp(-dist / 2) + 0.20 * np.exp(-dist / 0.4) + 0.32 * np.exp(-3.5 * (1 - xx @ xx.T))
        # A randomized empirical distribution gives independent panel seeds without extra answers.
        pop = 0.72 + 0.28 * rng.exponential(size=len(group))
        mean = kernel @ (pop / pop.sum())
        sources.append(
            dict(
                group=group,
                kernel=kernel,
                mean=mean,
                residual=mean.copy(),
                variance=kernel.diagonal().copy() + 0.35,
                factor=[],
                positions=[],
            )
        )
    counts = np.zeros(len(sources), int)
    sequence = []
    for n in range(1, args.probe_count + 1):
        j = int(np.argmax(mass * n - counts))
        p = sources[j]
        gain = p["residual"].clip(0) / np.sqrt(p["variance"].clip(0.01))
        values = p["positions"]
        gain[values] = -np.inf
        pick = int(gain.argmax())
        cross = p["kernel"][:, pick].copy()
        if p["factor"]:
            factor = np.stack(p["factor"], axis=1)
            cross -= factor @ factor[pick]
        pivot = np.sqrt(p["variance"][pick])
        col = cross / pivot
        p["residual"] -= col * p["residual"][pick] / pivot
        p["variance"] -= col**2
        p["factor"].append(col)
        values.append(pick)
        sequence.append(p["group"][pick])
        counts[j] += 1
    sequence = np.asarray(sequence)
    weights = np.ones(len(sequence))
    for j, p in enumerate(sources):
        keep = p["positions"]
        kernel = p["kernel"][np.ix_(keep, keep)]
        weight = np.linalg.solve(kernel + 0.35 * np.eye(len(keep)), p["mean"][keep])
        weight = np.clip(weight, 0.18 / len(keep), 3.2 / len(keep))
        weights[dom[sequence] == j] = weight / weight.mean()
    return sequence, observe(args.raw, ids[sequence], n_items), weights


def observe(raw, ids, n_items):
    found = np.full((112, len(ids)), np.nan, dtype=np.float32)
    lookup = np.full(n_items, -1, dtype=int)
    lookup[ids] = np.arange(len(ids))
    for frame in chunks(raw / "train.csv"):
        keep = lookup[frame.prompt_id.to_numpy()] >= 0
        rows = frame[keep].drop_duplicates(["model_id", "prompt_id"], keep="last")
        found[rows.model_id, lookup[rows.prompt_id]] = rows.label
    if not np.isfinite(found).all():
        raise ValueError("All candidates require a graded response to each selected question.")
    return found


def stratify(vectors, aux, text, dom):
    features, _ = question_map(vectors, text, aux)
    groups = np.empty(len(dom), int)
    labels = []
    for j in np.unique(dom):
        use = dom == j
        n = min(12, max(2, use.sum() // 27))
        km = KMeans(n_clusters=n, n_init=6, random_state=8087).fit(features[use])
        groups[use] = len(labels) + km.labels_
        labels.extend([j] * n)
    return groups, np.array(labels)


def survey_weights(groups, idx):
    n = np.bincount(groups)
    observed = np.bincount(groups[idx], minlength=len(n))
    return n[groups[idx]] / observed[groups[idx]]


def relevance(y, source, group, used, meta, size, mass):
    weights = survey_weights(group, used)
    ds = source[used]
    abil = posterior_abilities(y, ds, meta, size, dict(noise_corr=0.5, metadata=0.5, noise=1.0), weights)
    priorities = []
    for j in np.unique(source):
        ix = ds == j
        contenders = np.argsort(-abil[:, j])[:12]
        yj = y[contenders][:, ix]
        delta = yj[0] - yj[1:]
        sd = delta.std(1).clip(0.12)
        gap = abil[contenders[0], j] - abil[contenders[1:], j]
        likely = ndtr(-gap * np.sqrt(ix.sum() + 2) / sd)
        uncertainty = (sd * likely).sum() / len(likely)
        priorities.append((0.12 + uncertainty) * np.sqrt(0.12 / len(mass) + 0.88 * mass[j]))
    return np.array(priorities)


def acquire(args, seed, ids, x, dom, text, aux, meta, size, source_mass, n_items):
    rng = np.random.default_rng(seed + 8087)
    cells, cell_dom = stratify(x, aux, text, dom)
    population = np.bincount(cells)
    pools = [rng.permutation(np.flatnonzero(cells == i)) for i in range(len(population))]
    counts = np.zeros(len(population), int)
    used = []
    y = np.zeros((112, 0), dtype=np.float32)
    for stop in (100, 145, 175, 200):
        if not used:
            mass = 0.50 / len(source_mass) + 0.50 * np.sqrt(source_mass) / np.sqrt(source_mass).sum()
        else:
            mass = relevance(y, dom, cells, np.asarray(used), meta, size, source_mass)
            mass = 0.25 / len(mass) + 0.75 * mass / mass.sum()
        first = len(used)
        for n in range(first + 1, stop + 1):
            acquired = np.bincount(cell_dom, weights=counts, minlength=len(mass))
            j = int(np.argmax(np.where(acquired < np.bincount(dom), mass * n - acquired, -np.inf)))
            present = np.flatnonzero(cell_dom == j)
            # Cover each observed format before repeatedly querying the largest stratum.
            target = 0.20 / len(present) + 0.80 * population[present] / population[present].sum()
            wanting = target * (acquired[j] + 1) - counts[present]
            c = present[np.argmax(np.where(counts[present] < population[present], wanting, -np.inf))]
            used.append(pools[c][counts[c]])
            counts[c] += 1
        extra = np.asarray(used[first:])
        new = observe(args.raw, ids[extra], n_items)
        y = np.column_stack((y, new))
    idx = np.asarray(used)
    return idx, y, survey_weights(cells, idx)


def acquire_precision(args, seed, ids, x, dom, text, aux, meta, size, source_mass, n_items):
    from population_panel import population_coordinates

    rng = np.random.default_rng(seed + 5167)
    cells = np.empty(len(ids), int)
    cell_dom = []
    for j in np.unique(dom):
        keep = dom == j
        vec, _ = population_coordinates(x[keep], x[keep], aux[keep], aux[keep], text[keep], text[keep])
        count = min(5, max(2, keep.sum() // 42))
        clusters = KMeans(n_clusters=count, n_init=6, random_state=5167).fit_predict(vec)
        cells[keep] = len(cell_dom) + clusters
        cell_dom.extend([j] * count)
    cell_dom = np.asarray(cell_dom)
    pool = [rng.permutation(np.flatnonzero(cells == i)) for i in range(len(cell_dom))]
    pop = np.array(list(map(len, pool)))
    count = np.zeros(len(pop), int)
    source_weight = 0.50 / len(source_mass) + 0.50 * source_mass
    size_by_source = np.bincount(dom)
    used = []
    y = np.zeros((112, 0), np.float32)
    for stop in (74, 106, 140, 170, 200):
        allocation = source_weight[cell_dom] * pop / size_by_source[cell_dom]
        if not used:
            # Give each stratum an observed common question before estimating its comparison variance.
            for c in rng.permutation(len(pop)):
                used.append(pool[c][0])
                count[c] += 1
        else:
            idx = np.asarray(used)
            dd = dom[idx]
            weight = survey_weights(cells, idx)
            z = posterior_abilities(
                y, dd, meta, size, dict(noise_corr=0.59, metadata=0.45, noise=1.1), weight
            )
            for j in np.unique(dom):
                group = dd == j
                contenders = np.argsort(-z[:, j])[:16]
                pair = np.triu_indices(len(contenders), 1)
                strongest = z[contenders, j]
                temp = 0.039 * np.sqrt(18 / (group.sum() + 5))
                prob = softmax((strongest - strongest[0]) / temp)
                mix = prob[pair[0]] * prob[pair[1]]
                mix /= mix.sum()
                answers = y[contenders]
                disagreement = mix @ np.abs(answers[pair[0]] - answers[pair[1]])
                domain_mean = np.average(disagreement[group], weights=weight[group])
                opportunity = 0.45 + 0.55 * np.mean(
                    np.minimum(1, np.exp((strongest[1:] - strongest[0]) / (temp * 1.5)))
                )
                for c in np.flatnonzero(cell_dom == j):
                    take = cells[idx] == c
                    variation = (disagreement[take].sum() + 5 * domain_mean + 0.35) / (take.sum() + 6)
                    allocation[c] *= np.sqrt(variation + 0.02) * opportunity
        allocation /= allocation.sum()
        for n in range(len(used) + 1, stop + 1):
            c = int(np.argmax(np.where(count < pop, allocation * n - count, -np.inf)))
            used.append(pool[c][count[c]])
            count[c] += 1
        new = np.array(used[y.shape[1] :])
        y = np.column_stack((y, observe(args.raw, ids[new], n_items)))
    idx = np.array(used)
    return idx, y, survey_weights(cells, idx)
