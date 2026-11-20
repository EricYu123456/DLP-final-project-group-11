"""Use an ungraded question population to estimate conditional ability from a small shared survey."""

import numpy as np
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd

from panel_rasch import subtype_matrix


def population_coordinates(bank, x, bank_aux, aux, bank_text, text):
    parts = []
    for big, small, rank in ((bank, x, 5), (bank_aux, aux, 4)):
        mean = big.mean(0)
        scale = big.std(0).clip(0.06)
        _, _, axes = randomized_svd(
            (big - mean) / scale, min(rank, big.shape[1], len(big) - 1), n_iter=3, random_state=518
        )
        b = (big - mean) / scale @ axes.T
        q = (small - mean) / scale @ axes.T
        parts.append((b / b.std(0).clip(0.35), q / b.std(0).clip(0.35)))
    bt, tt = subtype_matrix(bank_text), subtype_matrix(text)
    keep = bt.std(0) > 0.10
    parts.append((bt[:, keep], tt[:, keep]))
    return tuple(np.column_stack(z) for z in zip(*parts))


def calibrated_weights(bank, x, bank_dom, dom, bank_aux, aux, bank_text, text, prior=None):
    output = np.ones(len(dom))
    for j in np.unique(dom):
        known, seen = bank_dom == j, dom == j
        b, q = population_coordinates(
            bank[known], x[seen], bank_aux[known], aux[seen], bank_text[known], text[seen]
        )
        q = (q - b.mean(0)).clip(-3, 3)
        base = np.ones(seen.sum()) if prior is None else prior[seen]
        base = base / base.sum()

        # The dual of entropy balancing. With few probes the ridge and finite weights retain
        # sampling efficiency while correcting imbalance in observable sources of difficulty.
        def objective(v):
            logits = np.log(base) + q @ v
            largest = logits.max()
            exp = np.exp(logits - largest)
            prob = exp / exp.sum()
            return np.log(exp.sum()) + largest + 0.22 * (v @ v), prob @ q + 0.44 * v

        solution = minimize(objective, np.zeros(q.shape[1]), jac=True, method="BFGS").x
        adj = np.exp(np.clip(q @ solution, -1.25, 1.25))
        value = base * adj
        val = 0.22 * base + 0.78 * value / value.sum()
        output[seen] = val * seen.sum()
    return output
