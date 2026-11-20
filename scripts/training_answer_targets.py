"""Match reference answers only for historical training prompts and resolve free-answer solver groups."""

import argparse
import csv
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import requests

from coldstart.answer_evidence import question_parts
from embed_new_model import read_prompts
from evaluate_capability_router import ROOT, input_context
from research.verify_source_items import norm


SOURCES = {
    "logi_test.txt": "lgw863/LogiQA-dataset/ff6c4cbca47627b3ac2da94a29fa28204a167b41/Test.txt",
    "MathQA.zip": "math-QA/math-QA/4f958013c164a40745a5b771dfb005df308deb4a/data/MathQA.zip",
    "truth.csv": "sylinrl/TruthfulQA/013686a06be7a7bde5bf8223943e106c7250123c/TruthfulQA.csv",
}


def files(dest):
    dest.mkdir(exist_ok=True)
    for name, origin in SOURCES.items():
        file = dest / name
        if not file.exists():
            response = requests.get("https://raw.githubusercontent.com/" + origin, timeout=60)
            response.raise_for_status()
            file.write_bytes(response.content)


def key(s):
    return "".join(norm(s).casefold().split())


def training_targets(args):
    dest = args.out / "source_answers"
    files(dest)
    d, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        tr = f["training"]
    ids = d.prompt_ids[tr]
    texts = read_prompts(ids, ROOT / "data/raw/question_order.csv")
    with np.load(args.out / "historical_answer_keys.npz") as f:
        assert np.array_equal(f["prompt_ids"], ids)
        options = f["option"].copy()
    logical, math, truth = {}, {}, {}
    for block in re.split(r"\n\s*\n", (dest / "logi_test.txt").read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) < 7 or lines[0].strip().lower() not in ("a", "b", "c", "d"):
            continue
        alternatives = [re.sub(r"^\s*[A-D][.)]\s*", "", s).strip() for s in lines[-4:]]
        label = ord(lines[0].strip().lower()) - ord("a")
        logical.setdefault((key("".join(lines[1:-5])), key(lines[-5])), set()).add(key(alternatives[label]))
    with zipfile.ZipFile(dest / "MathQA.zip") as archive:
        for v in json.load(archive.open("test.json")):
            parts = re.split(r"(?:^|,\s*)[a-e]\s*\)\s*", v["options"].strip(), flags=re.I)[1:]
            i = "abcde".find(str(v["correct"]).strip().lower())
            if len(parts) == 5 and i >= 0:
                math.setdefault(key(v["Problem"]), set()).add(norm(parts[i]))
    with (dest / "truth.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            truth[key(row["Question"])] = [
                s.strip() for s in (row["Best Answer"] + ";" + row["Correct Answers"]).split(";") if s.strip()
            ]
    free = {}
    for i, pid in enumerate(ids):
        stem, choices = question_parts(texts[int(pid)])
        source = d.benchmark_names[d.domains[tr[i]]]
        if source == "LogiQA":
            p = stem.removeprefix("Passage:").split("\nQuestion:", 1)
            gold = logical.get(tuple(map(key, p)), set())
            found = [j for j, c in enumerate(choices) if key(c) in gold]
            if len(gold) == 1 and len(found) == 1:
                options[i] = found[0]
        elif source == "MathQA":
            gold = math.get(key(stem.removeprefix("Question:")), set())
            if len(gold) == 1:
                free[int(pid)] = dict(source=source, answers=sorted(gold))
        elif source == "TruthfulQA":
            candidates = truth.get(key(stem))
            if candidates:
                free[int(pid)] = dict(source=source, answers=candidates)
    np.savez_compressed(dest / "training.npz", prompt_ids=ids, option=options)
    (dest / "free.json").write_text(json.dumps(free, ensure_ascii=False), encoding="utf-8")
    print("Historical option and free response keys", int((options >= 0).sum()), len(free), flush=True)


def numeric(s):
    clean = s.lower().replace(",", "").strip().removesuffix(".").strip()
    clean = re.sub(r"^(?:the answer is|answer[ :=]*|approximately|about)\s*", "", clean)
    m = re.fullmatch(
        r"\$?\s*([-+]?\d+(?:\s*\.\s*\d+)?(?:\s*/\s*\d+)?)\s*(?:%|cm|km|hours?|days?|minutes?|years?|kg|meters?|dollars?)?",
        clean,
    )
    if not m:
        return None
    n = m.group(1).replace(" ", "").split("/")
    if len(n) == 2 and float(n[1]) == 0:
        return None
    return float(n[0]) / (float(n[1]) if len(n) == 2 else 1)


def resolve(dest, ids, evidence, raw):
    with np.load(dest / "source_answers/training.npz") as f:
        lookup = {int(pid): i for i, pid in enumerate(f["prompt_ids"])}
        keys = f["option"][[lookup[int(pid)] for pid in ids]]
    free = json.loads((dest / "source_answers/free.json").read_text(encoding="utf-8"))
    normalize = lambda v: re.sub(r"[^\w]+", " ", norm(v).lower()).strip()
    for i, pid in enumerate(ids):
        row = free.get(str(pid))
        if not row or raw["mc"][i]:
            continue
        targets = {normalize(a) for a in row["answers"]}
        numbers = {n for a in row["answers"] if (n := numeric(a)) is not None}
        matches = set()
        definite = True
        for e, (p, words) in enumerate(evidence):
            if p.shape[-1] > 6 and not p[i, 0, 6]:
                continue
            val = numeric(str(words[i]))
            same = normalize(words[i]) in targets or (
                val is not None and any(np.isclose(val, n, rtol=1e-4) for n in numbers)
            )
            if same:
                matches.add(int(raw["votes"][i, e].argmax()))
            if val is None:
                definite = False
        if len(matches) == 1:
            keys[i] = next(iter(matches))
        elif not matches and row["source"] == "MathQA" and definite and numbers:
            keys[i] = int(raw["mask"][i].sum() - 1)
    return keys


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    training_targets(p.parse_args())
