"""Set aside fresh question groups for a second evaluation of the same unseen model families."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evaluate_capability_router import ProbeBank, ROOT, input_context
from run_experiment import atomic_json


def prepare(args):
    d, s, cfg, _, prepared, _ = input_context(args.run)
    bank = ProbeBank(d, s, cfg["seeds"])
    old_probes = np.unique(np.concatenate(list(bank.orders.values())))
    omit = set(d.group_ids[bank.items[old_probes]].tolist())
    meta = pd.read_csv(prepared / "prompts.csv", usecols=["prompt_id", "group_id", "original_split"])
    # Preserve the cold-start design. Neither the old router's test items nor any public
    # release evaluation question group can become a second confirmation question.
    omit.update(meta.loc[meta.original_split != "train", "group_id"].unique().tolist())
    for pack in args.root.glob("release112*/panel_*.npz"):
        with np.load(pack) as panel:
            omit.update(meta.loc[meta.prompt_id.isin(panel["prompt_ids"]), "group_id"].tolist())
    eligible = bank.items[~np.isin(d.group_ids[bank.items], list(omit))]
    rng = np.random.default_rng(20261120)
    sizes = np.bincount(d.domains[bank.items])
    exact = args.count * sizes / sizes.sum()
    counts = np.floor(exact).astype(int)
    counts[np.argsort(-(exact % 1))[: args.count - counts.sum()]] += 1
    groups = []
    for dom, count in enumerate(counts):
        ix = eligible[d.domains[eligible] == dom]
        unique, freq = np.unique(d.group_ids[ix], return_counts=True)
        order = rng.permutation(len(unique))
        groups.extend(unique[order[: np.searchsorted(np.cumsum(freq[order]), count) + 1]])
    target = bank.items[np.isin(d.group_ids[bank.items], groups)]
    train = bank.items[~np.isin(d.group_ids[bank.items], groups)]
    obj = dict(
        training=train,
        confirmation=target,
        development=np.asarray(s.val_items),
        previous_test=np.asarray(s.test_items),
    )
    for seed in bank.seeds:
        obj[f"balanced_{seed}"] = bank.items[bank.orders["balanced", seed]]
    np.savez(args.out / "questions.npz", **obj)
    atomic_json(
        args.report / "design.json",
        dict(
            seed=20261120,
            training_questions=len(train),
            confirmation_questions=len(target),
            development_questions=len(s.val_items),
            count_by_benchmark=dict(zip(d.benchmark_names, map(int, np.bincount(d.domains[target])))),
            candidate_probe_cap=200,
            primary_roster="audited_13",
            secondary_roster="expanded_37",
            endpoints=["macro_accuracy", "micro_accuracy", "top_model_share"],
            source="Question groups from the earlier historical training bank, excluding previous probes and official release validation/test groups. All supervised shared models and controls in this confirmation study are retrained without these groups.",
        ),
    )
    print(
        f"Set aside {len(target)} questions from {len(groups)} groups. {len(train)} historical training questions remain.",
        flush=True,
    )


def references(args):
    from train_embedllm_reference import fit
    import torch

    torch.set_num_threads(1)
    d, s, _, _, prepared, family = input_context(args.run)
    with np.load(args.out / "questions.npz") as pack:
        train = pack["training"]
    if args.phase == "refit-reference":
        rows = np.r_[s.train_models, s.val_models]
        fit(
            d.embeddings[train],
            d.labels[np.ix_(rows, train)],
            d.model_ids[rows],
            d.prompt_ids[train],
            args.out,
            "historical75",
        )
        return
    fit(
        d.embeddings[train],
        d.labels[np.ix_(s.train_models, train)],
        d.model_ids[s.train_models],
        d.prompt_ids[train],
        args.out,
        "historical",
    )
    from evaluate_metadata_excluded import load_oracle

    extra = load_oracle(d, prepared / "embedllm.npz", family)
    mids = np.r_[d.model_ids, extra.model_ids]
    labels = np.vstack((d.labels[:, train], extra.labels[:, train]))
    ix = np.argsort(mids)
    fit(d.embeddings[train], labels[ix], mids[ix], d.prompt_ids[train], args.out, "known_ids")


def previous_router(args):
    from types import SimpleNamespace
    import torch
    from coldstart.models import create_decoder, save_decoder
    from pilot_routing_learning import B, VARIANT, checkpoint_eval, fit_trajectory
    from run_experiment import decoder_configuration, training_configuration

    torch.set_num_threads(2)
    d, s, cfg, strength, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr, dev = f["training"], f["development"]
        panels = {seed: f[f"balanced_{seed}"] for seed in cfg["seeds"]}
    target = d.subset(s.val_models, dev)
    result = []
    for seed, ids in panels.items():
        support = d.subset(s.val_models, ids)
        epoch = 0
        for hist in (s.train_models, np.r_[s.train_models, s.val_models]):
            cohort = len(hist)
            options = training_configuration(cfg, seed, "cpu")
            split = SimpleNamespace(train_models=hist, protocol=s.protocol)
            spec = decoder_configuration(d, split, cfg, VARIANT)
            if cohort == len(s.train_models):
                params = (support, target, cfg, strength)
                evaluation = checkpoint_eval
            else:
                options.epochs = epoch
                options.selection_interval = epoch
                order = iter((1.0, 2.0))
                params = ()

                def evaluation(_):
                    return next(order), 0.0

            best, _ = fit_trajectory(spec, options, d.subset(hist, tr), params, evaluation, False)
            chosen = best[B]
            epoch = chosen["epoch"]
            dec = create_decoder(spec, seed)
            dec.load_state_dict(chosen["state"])
            fields = dict(
                seed=seed,
                reference_models=cohort,
                epoch=epoch,
                training_questions=len(tr),
                prior_strength=strength,
            )
            save_decoder(args.out / f"previous8_{cohort}" / f"{B}_{seed}.pt", dec, fields)
            result.append(fields)
            print("Refit original 8D", fields, flush=True)
    atomic_json(args.report / "previous8_refit.json", dict(training=result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase", choices=("prepare", "references", "refit-reference", "previous-8d"), required=True
    )
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--root", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--report", type=Path, default=ROOT / "reports/routing/capability/confirmation")
    p.add_argument("--count", type=int, default=5000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=3):
        {
            "prepare": prepare,
            "references": references,
            "refit-reference": references,
            "previous-8d": previous_router,
        }[args.phase](args)
