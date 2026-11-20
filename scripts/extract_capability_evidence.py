"""Extract factual answer evidence from independent proxy LMs, without candidate answers."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pilot_routing_learning import ROOT, input_context
from coldstart.answer_evidence import answer_reader, expert_gate, selective_answers, GATE_PROXIES, PROXIES
from train_capability_router import VramMeter, prompt_texts


def run(args):
    d, _, *_ = input_context(ROOT / "artifacts/main_v2")
    with np.load(args.out / "questions.npz") as split:
        parts = (
            ("training", "development") if args.phase == "development" else ("confirmation", "previous_test")
        )
        positions = np.concatenate([split[key] for key in parts])
    ids = d.prompt_ids[positions]
    texts = prompt_texts(ROOT / "data/raw/question_order.csv", ids)
    name = args.proxy + ("-gated" if args.gated else "")
    if args.gated:
        from train_conditional_router import proxy_data

        gate = expert_gate(dict(zip(GATE_PROXIES, proxy_data(args.out, ids, GATE_PROXIES, args.phase))),
                           multiple_only=args.proxy == "option-math")
        print(f"{name} will run on {gate.sum()} of {len(gate)} questions", flush=True)
    meter = VramMeter(8 if args.proxy in ("smol", "math", "option-math") else 16)
    start = perf_counter()
    try:
        if args.axes:
            from coldstart.skill_axes import SkillAxes

            model = SkillAxes(args.proxy)
        else:
            model = answer_reader(args.proxy)
        features, answers = [], []
        batch = {
            "smol": 64,
            "math": 32,
            "qwen": 32,
            "phi": 24,
            "yi": 24,
            "zephyr": 24,
            "bagel": 8,
            "moe": 8,
            "una": 4,
            "option-math": 16,
            "koala": 8,
        }[args.proxy]
        resume = args.proxy in ("bagel", "moe", "una", "option-math", "koala")
        for i in range(0, len(texts), 512):
            part = args.out / f"{name}_chunks" / f"{args.phase}_{i}.npz"
            if args.axes:
                p = model.encode_axes(texts[i : i + 512], batch)
                label = []
            elif resume and part.exists():
                with np.load(part) as f:
                    p, label = f["probabilities"], f["answers"]
            else:
                if args.gated:
                    p, label = selective_answers(model, texts[i : i + 512], gate[i : i + 512], batch)
                else:
                    p, label = model.encode(texts[i : i + 512], batch)
                if resume:
                    part.parent.mkdir(exist_ok=True)
                    np.savez_compressed(part, probabilities=p, answers=label)
            features.append(p)
            answers.append(label)
            if i % 2048 == 0:
                print(
                    f"{args.proxy} {min(i + 512, len(texts))}/{len(texts)}, {perf_counter() - start:.1f}s",
                    flush=True,
                )
        kind = "axes" if args.axes else "evidence"
        path = args.out / f"{kind}_{name}_{args.phase}.npz"
        np.savez_compressed(
            path,
            prompt_ids=ids,
            probabilities=np.vstack(features),
            answers=np.concatenate(answers),
            seconds=perf_counter() - start,
            board_peak_mib=meter.peak,
        )
        print(path, f"{perf_counter() - start:.1f}s, {meter.peak} MiB", flush=True)
    finally:
        if "model" in locals() and hasattr(model, "close"):
            model.close()
        meter.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    p.add_argument("--phase", choices=("development", "evaluation"), required=True)
    p.add_argument("--proxy", choices=[*PROXIES, "bagel", "moe", "una", "option-math", "koala"], required=True)
    p.add_argument("--axes", action="store_true")
    p.add_argument("--gated", action="store_true")
    torch.set_num_threads(1)
    run(p.parse_args())
