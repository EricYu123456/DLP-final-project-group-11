"""Train under nested RTX 5070 Ti wall-time and dedicated-memory budgets."""

import argparse
import csv
import gc
import subprocess
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from pilot_routing_learning import ROOT, input_context
from run_experiment import atomic_json
from coldstart.capability_text import ResponseEncoder, tokenize_prompts, tokenizer_for


def device_usage_query(utilization=False):
    fields = "memory.used,utilization.gpu" if utilization else "memory.used"
    return ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits", "-i", "0"]


class VramMeter:
    """Track total dedicated board use, including the Windows desktop and the CUDA context."""

    def __init__(self, cap):
        torch.cuda.init()
        self.initial = int(subprocess.check_output(device_usage_query(), text=True).strip())
        self.peak = self.initial
        self.active_samples = []
        total = torch.cuda.get_device_properties(0).total_memory
        self.limit = min(cap * 2**30, total) - (self.initial + 360) * 2**20
        torch.cuda.set_per_process_memory_fraction(self.limit / total)
        self.proc = subprocess.Popen(
            device_usage_query(True) + ["-lms", "500"],
            stdout=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._collect, daemon=True).start()

    def _collect(self):
        for line in self.proc.stdout:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and all(field.isdigit() for field in fields):
                used, utilization = map(int, fields)
                self.peak = max(self.peak, used)
                if utilization >= 50:
                    self.active_samples.append((used, utilization))

    def close(self):
        self.proc.terminate()
        self.proc.wait()


def prompt_texts(path, ids):
    wanted = set(map(int, ids))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        found = {
            int(r["prompt_id"]): r["prompt"] for r in csv.DictReader(stream) if int(r["prompt_id"]) in wanted
        }
    return [found[int(i)] for i in ids]


def create_member(spec, seed):
    torch.manual_seed(seed)
    net = ResponseEncoder(spec["size"], spec["n_models"], spec["n_domains"], spec["tune_blocks"]).cuda()
    optimizer = torch.optim.AdamW(
        [
            {"params": net.readout.parameters(), "lr": 0.0006},
            {
                "params": [p for p in net.backbone.parameters() if p.requires_grad],
                "lr": 0.000015 if spec["size"] == "small" else 0.000009,
            },
        ],
        weight_decay=0.012,
        fused=True,
    )
    return net, optimizer


def train_step(net, optimizer, ids, lengths, labels, domains, index):
    width = int(lengths[index].max())
    logits, predicted_domains = net(ids[index, :width], lengths[index])
    target = labels[index]
    prob = logits.sigmoid()
    # Emphasize predictable differences between LLMs on the same question.
    contrast = (prob - prob.mean(1, keepdim=True)) - (target - target.mean(1, keepdim=True))
    loss = (
        F.binary_cross_entropy_with_logits(logits, target)
        + 2 * contrast.square().mean()
        + 0.08 * F.cross_entropy(predicted_domains, domains[index])
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0, foreach=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def choose_batch(net, optimizer, data, limit):
    ids, lengths, _, _ = data
    long_rows = torch.where(lengths == ids.shape[1])[0]
    # Two real updates measure activation bytes per full-length example; both train on historical rows.
    for n in (4, 8, 16):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        train_step(net, optimizer, *data, long_rows[:n])
        torch.cuda.synchronize()
        if n == 8:
            lower = torch.cuda.max_memory_allocated()
    upper = torch.cuda.max_memory_allocated()
    per_row = max((upper - lower) / 8, 1)
    fixed = upper - 16 * per_row
    return max(4, int((limit * 0.96 - fixed) / per_row) // 4 * 4)


def run(args, meter, start):
    cap = args.cap
    size, limit, n_blocks = ("small", 192, 4) if cap == 8 else ("large", 256, 3)
    milestones = [2.2, 4.6, 8.1, 20, 35.5, 51, 67.5] if cap == 8 else [15, 35.5, 53, 67.5]
    if args.until is not None:
        milestones = [m for m in milestones if m <= args.until]
        if args.until not in milestones:
            milestones.append(args.until)
    d, s, _, _, _, _ = input_context(args.run)
    bank, val = np.asarray(s.train_items), np.asarray(s.val_items)
    labels = d.labels[np.ix_(s.train_models, bank)].T.copy()
    tokenizer = tokenizer_for(size)
    text_source = ROOT / "data/raw/question_order.csv"
    ids, lengths = tokenize_prompts(tokenizer, prompt_texts(text_source, d.prompt_ids[bank]), limit)
    val_tokens = tokenize_prompts(tokenizer, prompt_texts(text_source, d.prompt_ids[val]), limit)
    domain = d.benchmark_ids[bank]
    data = (ids.cuda(), lengths.cuda(), torch.from_numpy(labels).cuda(), torch.from_numpy(domain).cuda())
    spec = {
        "size": size,
        "limit": limit,
        "tune_blocks": n_blocks,
        "n_models": labels.shape[1],
        "n_domains": len(d.benchmark_names),
        "model_ids": d.model_ids[s.train_models].tolist(),
        "benchmark_names": list(d.benchmark_names),
    }
    dest = args.output / f"gpu{cap}"
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(401)
    sample_weights = 0.5 / len(domain) + 0.5 / (spec["n_domains"] * np.bincount(domain)[domain])
    net, optimizer = create_member(spec, 401)
    batch = choose_batch(net, optimizer, data, meter.limit)
    member, updates, examples = 1, 0, 28
    initial_batch = batch
    ledger = []
    print(f"{size}: {batch} training questions/update, {meter.initial} MiB before model load", flush=True)
    for milestone in milestones:
        net.train()
        while perf_counter() - start < milestone * 60 or not updates:
            indices = rng.choice(len(domain), size=batch, p=sample_weights)
            index = torch.as_tensor(indices, device="cuda")
            try:
                loss = train_step(net, optimizer, *data, index)
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if batch <= 4:
                    raise
                batch = max(4, ((batch * 4 // 5) // 4) * 4)
                print(f"Retrying with batch {batch} under the allocator limit", flush=True)
                continue
            updates += 1
            examples += len(index)
            if updates % 100 == 0:
                print(f"{perf_counter() - start:.0f}s, {updates} updates, loss {loss.item():.4f}", flush=True)
        torch.cuda.synchronize()
        name = "minute_" + format(milestone, "g").replace(".", "_")
        net.save_tuned(dest / f"{name}.pt", spec)
        scores, memberships = net.predict(*val_tokens, batch=batch)
        np.savez(
            dest / f"{name}_validation.npz",
            historical_scores=scores,
            domains=memberships,
            prompt_ids=d.prompt_ids[val],
        )
        row = {
            "stage": name,
            "member": member,
            "seed": 400 + member,
            "updates": updates,
            "training_examples": examples,
            "batch": batch,
            "proposed_batch": initial_batch,
            "max_tokens": limit,
            "elapsed_seconds": perf_counter() - start,
            "allocator_limit_mib": meter.limit / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "board_peak_mib": meter.peak,
            "board_initial_mib": meter.initial,
            "gpu": torch.cuda.get_device_name(0),
        }
        if meter.active_samples:
            active = np.asarray(meter.active_samples)
            row["median_active_board_mib"] = float(np.median(active[:, 0]))
            row["median_active_gpu_utilization"] = float(np.median(active[:, 1]))
        ledger.append(row)
        atomic_json(dest / "training.json", ledger)
        print(row, flush=True)
        if milestone in (8.1, 35.5) and milestone != milestones[-1]:
            del net, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            member += 1
            net, optimizer = create_member(spec, 400 + member)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/capability_routing")
    p.add_argument("--cap", type=int, choices=(8, 16), required=True)
    p.add_argument("--until", type=float, help="Stop after this many elapsed minutes of this trajectory.")
    args = p.parse_args()
    start = perf_counter()
    torch.set_num_threads(1)
    meter = VramMeter(args.cap)
    try:
        run(args, meter, start)
    finally:
        meter.close()


if __name__ == "__main__":
    main()
