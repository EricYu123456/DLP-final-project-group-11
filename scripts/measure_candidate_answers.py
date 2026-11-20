"""Measure a candidate's answer distribution once per question on a fixed onboarding panel."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from coldstart.gguf_evidence import GGUFAnswerEvidence
from evaluate_capability_router import ROOT, input_context
from panel_capabilities import panel_text


CANDIDATES = {
    "luminex": (
        "ConvexAI__Luminex-34B-v0.1",
        "mradermacher/Luminex-34B-v0.1-i1-GGUF",
        "9aca846ee91959176909d6f8c62b1d08e31da3a1",
        "Luminex-34B-v0.1.i1-IQ3_XXS.gguf",
    ),
    "tulu": (
        "TheBloke__tulu-30B-fp16",
        "TheBloke/tulu-30B-GGUF",
        "c95b903f0d48f96769ca5de2fa0287d5a1a75e1a",
        "tulu-30b.Q3_K_S.gguf",
    ),
    "hermes": (
        "NousResearch__Nous-Hermes-13b",
        "TheBloke/Nous-Hermes-13B-GGUF",
        "1fdc990e0588a21093ba3091f7117c3632e42901",
        "Nous-Hermes-13B.Q4_K_M.gguf",
    ),
    "medchat": (
        "AdaptLLM__medicine-chat",
        "TheBloke/medicine-chat-GGUF",
        "4b698781e153498a28f456df766e8cd3c2f22212",
        "medicine-chat.Q4_K_M.gguf",
    ),
    "medbase": (
        "AdaptLLM__medicine-LLM",
        "TheBloke/medicine-LLM-GGUF",
        "1765d37d0c391bbfbe529c8703da90fb2e5da287",
        "medicine-llm.Q4_K_M.gguf",
    ),
    "biomistral": (
        "BioMistral__BioMistral-7B",
        "MaziyarPanahi/BioMistral-7B-GGUF",
        "ae4f07544f1015dc8f5bf382b7582852638cbecf",
        "BioMistral-7B.Q4_K_M.gguf",
    ),
    "phi1": (
        "microsoft__phi-1_5",
        "TKDKid1000/phi-1_5-GGUF",
        "004eab50a07f5abcd7fbab443436332b08b23c07",
        "phi-1_5-Q4_K_M.gguf",
    ),
    "vicuna": (
        "lmsys__vicuna-7b-v1.5",
        "TheBloke/vicuna-7B-v1.5-GGUF",
        "8b4a138d6ba32660c42b5df6dad7ad5c23b80c8c",
        "vicuna-7b-v1.5.Q4_K_M.gguf",
    ),
    "codellama": (
        "codellama__CodeLlama-34b-Instruct-hf",
        "TheBloke/CodeLlama-34B-Instruct-GGUF",
        "7b84402c234acb1c5be542b5ecfc820ea3b74422",
        "codellama-34b-instruct.Q3_K_S.gguf",
    ),
    "gemma": (
        "google__gemma-2b-it",
        "mradermacher/gemma-2b-it-GGUF",
        "296aa97ccfdbbfa932f44ad339724ef0c5a9ff06",
        "gemma-2b-it.Q4_K_M.gguf",
    ),
    "pygmalion": (
        "Neko-Institute-of-Science__pygmalion-7b",
        "RichardErkhov/Neko-Institute-of-Science_-_pygmalion-7b-gguf",
        "c53ac3e5fdfbe50c4892634ea59a5a698f53d655",
        "pygmalion-7b.Q4_K_M.gguf",
    ),
    "openbuddy": (
        "OpenBuddy__openbuddy-codellama2-34b-v11.1-bf16",
        "TheBloke/openbuddy-llama2-34b-v11.1-bf16-GGUF",
        "dfaf85268b925453175ffab73806013e0c197667",
        "openbuddy-llama2-34b-v11.1-bf16.Q3_K_S.gguf",
    ),
    "awq70": (
        "TheBloke__CodeLlama-70B-Instruct-AWQ",
        "TheBloke/CodeLlama-70B-Instruct-AWQ",
        "120b7068c3e2f17f9c678ad732390aae6ad508ba",
        "model.safetensors.index.json",
    ),
}


def measure(args):
    d, *_ = input_context(args.run)
    with np.load(args.out / "questions.npz") as f:
        ids = d.prompt_ids[f["balanced_11"][:200]]
    texts = panel_text(ids, ROOT / "data/raw")
    name, repo, rev, filename = CANDIDATES[args.candidate]
    source = args.gguf or ROOT / "tmp/llama-b12550" / filename
    dest = args.out / "measured_candidates"
    dest.mkdir(exist_ok=True)
    outfile = dest / f"{args.candidate}.npz"
    if outfile.exists():
        with np.load(outfile) as f:
            if np.array_equal(f["prompt_ids"], ids) and str(f["model_name"]) == name:
                return
    start = perf_counter()
    extra = {}
    if args.candidate == "awq70":
        from measure_awq_panel import one_token_panel

        prob, answers, extra = one_token_panel(args.awq_dir, texts)
    else:
        model = GGUFAnswerEvidence(args.candidate, False, source, cpu=args.cpu)
        try:
            prob, answers = model.encode_panel(texts)
        finally:
            model.close()
    np.savez_compressed(
        outfile,
        prompt_ids=ids,
        model_name=name,
        repo=repo,
        revision=rev,
        filename=filename,
        probabilities=prob,
        answers=answers,
        seconds=perf_counter() - start,
        device="cpu" if args.cpu else "cuda",
        **extra,
    )
    print(name, len(ids), "actual model calls", round(perf_counter() - start, 1), "seconds", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", choices=CANDIDATES, required=True)
    p.add_argument("--gguf", type=Path)
    p.add_argument("--awq-dir", type=Path, default=ROOT / "tmp/awq70")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--run", type=Path, default=ROOT / "artifacts/main_v2")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/capability_routing/confirmation")
    measure(p.parse_args())
