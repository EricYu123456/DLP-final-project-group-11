"""Read query-only answer evidence from a local quantized GGUF auxiliary model."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
import requests
from huggingface_hub import hf_hub_download

from .answer_evidence import question_parts


SOURCES = {
    "koala": (
        "mradermacher/koala-13B-HF-GGUF",
        "0864781a6797fbdc6834fff3613c20cc045528ab",
        "koala-13B-HF.Q4_K_M.gguf",
    ),
    "una": (
        "mradermacher/UNA-SimpleSmaug-34b-v1beta-i1-GGUF",
        "a1cfd099aa3ed10abfe002cc7d97908d0ab088e4",
        "UNA-SimpleSmaug-34b-v1beta.i1-IQ3_XXS.gguf",
    ),
    "bagel": (
        "dranger003/bagel-34b-v0.2-iMat.GGUF",
        "2b8354524c2da355082381365557948269d75d15",
        "ggml-bagel-34b-v0.2-iq3_xxs.gguf",
    ),
    "moe": (
        "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
        "eea7b2be5805a5f151f8847ede8e5f9a9284bf77",
        "Qwen3-30B-A3B-Instruct-2507-Q3_K_S.gguf",
    ),
}


class GGUFAnswerEvidence:
    def __init__(self, name, published, model_file=None, cpu=False):
        self.name = name
        self.published = published
        default = Path(__file__).resolve().parents[2] / "tmp/llama-b12550"
        if model_file is None:
            repo, revision, filename = SOURCES[name]
            local = Path(os.environ.get(f"{name.upper()}_GGUF", default / filename))
            model = str(local) if local.is_file() else hf_hub_download(repo, filename, revision=revision)
        else:
            model = str(Path(model_file).resolve(strict=True))
        dense = name in ("bagel", "una")
        layers = (
            os.environ.get("ROUTING_DENSE_GPU_LAYERS", "60" if name == "una" else "55") if dense else "auto"
        )
        self.slots = 4 if model_file is not None or dense else 8
        exe = os.environ.get("LLAMA_SERVER") or next(
            (str(p) for p in default.rglob("llama-server.exe")), None
        )
        if not exe:
            raise FileNotFoundError("Set LLAMA_SERVER to the llama.cpp CUDA server executable.")
        with socket.socket() as local:
            local.bind(("127.0.0.1", 0))
            port = local.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        cmd = [
            exe,
            "-m",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-ngl",
            "0" if cpu else layers,
            "--fit-target",
            "1900",
            "-np",
            str(self.slots),
            "-c",
            {"bagel": "2048", "koala": "8192"}.get(name, "4096"),
            "-b",
            "256" if dense else "512",
            "-ub",
            "96" if dense else "192",
            "-fa",
            "on",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--cache-ram",
            "0",
            "--no-cache-prompt",
            "--no-webui",
        ]
        if cpu:
            cmd.extend(["--device", "none", "-t", "4"])
        default.mkdir(parents=True, exist_ok=True)
        self.log = (default / f"{name}-server.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(240):
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited with status {self.process.returncode}")
                try:
                    if requests.get(self.url + "/health", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f"The {name} auxiliary model did not finish loading in two minutes.")
            self.token_ids = {}
            for c in "ABCD":
                for label in (c, " " + c):
                    tokens = self.post("/tokenize", dict(content=label, add_special=False))["tokens"]
                    self.token_ids[int(tokens[-1])] = "ABCD".index(c)
        except BaseException:
            self.close()
            raise

    def post(self, path, body):
        response = requests.post(self.url + path, json=body, timeout=150)
        response.raise_for_status()
        return response.json()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=25)
        self.log.close()

    def query(self, text, options, order=None):
        if len(text) > 1240:
            text = text[:230] + "\n" + text[-1010:]
        if order is None:
            prompt = text + "\nGive only the final answer, as briefly as possible."
        else:
            formatted = "\n".join(f"{letter}. {options[i][:250]}" for letter, i in zip("ABCD", order))
            prompt = text + "\n" + formatted + "\nReply with just the correct option letter."
        if self.name == "bagel":
            prompt = "USER: " + prompt + "\nASSISTANT:"
        else:
            prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        return self.completion(prompt, order, 480, 8)

    def completion(self, prompt, order, limit, free_tokens):
        if len(prompt) > 850:
            ids = self.post("/tokenize", dict(content=prompt, add_special=True, parse_special=True))["tokens"]
            if len(ids) > limit:
                prompt = ids[:96] + ids[-(limit - 96) :]
        data = dict(
            prompt=prompt,
            n_predict=1 if order is not None else free_tokens,
            temperature=-1,
            n_probs=12 if order is not None else 1,
            cache_prompt=False,
            stop=["<|im_end|>", "\n"],
        )
        if order is not None:
            data["logit_bias"] = [[key, 8.0] for key in self.token_ids]
        result = self.post("/completion", data)
        entries = result.get("completion_probabilities", result.get("probs", []))
        info = np.zeros(6, dtype=np.float32)
        if not entries:
            return info, result.get("content", "").strip()
        if order is not None:
            mass = np.full(4, 1e-10)
            for token in entries[0].get("top_logprobs", []):
                slot = self.token_ids.get(token.get("id"))
                if slot is not None:
                    mass[slot] += np.exp(token["logprob"])
            info[:4] = (mass / mass.sum())[np.argsort(order)]
            info[4:] = np.log(mass.sum()), np.log(mass.max())
        else:
            logp = [r.get("logprob", 0.0) for r in entries]
            info[4:] = logp[0], np.mean(logp)
        return info, result.get("content", "").strip()

    def encode_panel(self, texts):
        """Submit each original onboarding prompt exactly once in its published order and format."""

        def probe(text):
            _, options = question_parts(text)
            return self.completion(str(text), [0, 1, 2, 3] if options else None, 990, 16)

        with ThreadPoolExecutor(max_workers=self.slots) as pool:
            result = list(pool.map(probe, texts))
        return np.stack([v[0] for v in result]), np.asarray([v[1] for v in result])

    def single(self, text):
        stem, options = question_parts(text, self.published)
        if not options:
            info, answer = self.query(stem, [])
            return np.stack((info, np.zeros_like(info))), answer
        first, _ = self.query(stem, options, [0, 1, 2, 3])
        second = self.query(stem, options, [2, 0, 3, 1])[0] if first[:4].max() < 0.85 else first
        return np.stack((first, second)), ""

    def encode(self, texts, batch=8):
        if self.name in ("una", "koala"):
            probabilities, answers = self.encode_panel(texts)
            return np.repeat(probabilities[:, None, :], 2, axis=1), answers
        with ThreadPoolExecutor(max_workers=min(batch, self.slots)) as pool:
            results = list(pool.map(self.single, texts))
        return np.stack([r[0] for r in results]), np.array([r[1] for r in results])
