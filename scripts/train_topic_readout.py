"""Train a CPU decoder for LLM-specific strengths within semantic question clusters."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

from embed_new_model import read_prompts
from evaluate_capability_router import ROOT, ProbeBank, get_kernel, input_context
from coldstart.topic_readout import features, fit_topics
from run_experiment import atomic_json


def run(args):
    start = perf_counter()
    d, s, cfg, *_ = input_context(args.run)
    bank = ProbeBank(d, s, cfg["seeds"])
    kernel = get_kernel(args, d, s)
    val = np.asarray(s.val_items)
    all_ids = d.prompt_ids[np.r_[bank.items, val]]
    lookup = read_prompts(all_ids, ROOT / "data/raw/question_order.csv")
    texts = [lookup[i] for i in all_ids]
    n = len(bank.items)
    x = d.embeddings[bank.items]
    pack = fit_topics(x, texts[:n], bank.domains)
    design = features(pack, x, texts[:n])
    val_x = features(pack, d.embeddings[val], texts[n:])
    _, mem = kernel.predict(d.embeddings[val])
    residual = bank.history.T - bank.domain_means[bank.domains]
    residual -= residual.mean(1, keepdims=True)
    axes = bank.basis.axes[:, :16] * np.sqrt(len(bank.history))
    coords = residual @ axes
    scale = coords.std(0).clip(0.05)
    weights = np.sqrt(n / (bank.basis.domain_count * np.bincount(bank.domains)[bank.domains]))
    out = args.weights / "topic"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "topic_fit.npz", **pack, domain_means=bank.domain_means, model_axes=axes, scale=scale)
    elapsed = {}
    model = None
    for total, extra in ((160, 160), (420, 260)):
        next_model = CatBoostRegressor(
            iterations=extra,
            depth=5,
            learning_rate=0.06,
            loss_function="MultiRMSE",
            thread_count=4,
            verbose=False,
            random_seed=471,
            allow_writing_files=False,
        )
        next_model.fit(design, coords / scale, sample_weight=weights, init_model=model)
        model = next_model
        stage = f"trees_{total}"
        model.save_model(str(out / f"{stage}.cbm"))
        p = mem @ bank.domain_means + (model.predict(val_x, thread_count=2) * scale) @ axes.T
        np.savez(
            out / f"{stage}_validation.npz", historical_scores=p, domains=mem, prompt_ids=d.prompt_ids[val]
        )
        elapsed[stage] = perf_counter() - start
        atomic_json(out / "times.json", elapsed)
        print(stage, elapsed[stage], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    parser.add_argument("--weights", type=Path, default=ROOT / "artifacts/capability_routing")
    parser.set_defaults(phase="validation")
    with threadpool_limits(limits=3):
        run(parser.parse_args())
