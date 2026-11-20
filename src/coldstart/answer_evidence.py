"""Predict question difficulty from independent small LMs' answers and option-order stability."""

import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROXIES = {
    "smol": ("HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"),
    "math": ("Qwen/Qwen2.5-Math-1.5B-Instruct", "aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"),
    "qwen": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "phi": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"),
    "yi": ("01-ai/Yi-1.5-6B-Chat", "771924d1c83d67527d665913415d7086f11ea9c0"),
    "zephyr": ("HuggingFaceH4/zephyr-7b-alpha", "014792bbb59d04ced3b5a9b8b4dfc926655d958f"),
}


def answer_reader(name, published=False):
    if name == "option-math":
        from .option_likelihood import OptionLikelihood

        return OptionLikelihood(published)
    if name in ("bagel", "moe", "una", "una-mc", "koala"):
        if (name.startswith("una") or name == "koala") and published:
            raise ValueError(f"{name} is itself a candidate in the official 112-model roster.")
        from .gguf_evidence import GGUFAnswerEvidence

        return GGUFAnswerEvidence(name.removesuffix("-mc"), published)
    return AnswerEvidence(name, published=published)


GATE_PROXIES = ("qwen", "math")


def expert_gate(evidence, multiple_only=False):
    from .latent_answers import answer_signature

    (left, la), (right, ra) = (evidence[name] for name in GATE_PROXIES)
    lp, rp = left[:, :, :4].mean(1), right[:, :, :4].mean(1)
    multiple = lp.sum(1) > 0.5
    switch_order = left[:, 0, :4].argmax(1) != left[:, 1, :4].argmax(1)
    disagree = lp.argmax(1) != rp.argmax(1)
    use = (lp.max(1) < 0.63) | switch_order | (disagree & (lp.max(1) < 0.94))
    if multiple_only:
        return multiple & use
    for i in np.flatnonzero(~multiple):
        a, aw = answer_signature(la[i])
        b, bw = answer_signature(ra[i])
        same = (a == b and bool(a)) if (a or b) else bool(aw and bw and len(aw & bw) / len(aw | bw) >= 0.65)
        use[i] = not same
    return use


def selective_answers(model, texts, mask, batch):
    """The seventh channel records whether the expensive solver actually ran."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros((len(mask), 2, 7), np.float32)
    ans = [""] * len(mask)
    if mask.any():
        use = np.flatnonzero(mask)
        prob, words = model.encode([texts[i] for i in use], batch)
        out[use, :, :6] = prob
        out[use, :, 6] = 1
        for i, word in zip(use, words):
            ans[i] = word
    return out, np.asarray(ans)


def encode_evidence(names, texts, batch, published=False, known=None):
    import gc

    needed = set(names)
    gated = sorted(v for v in needed if v.endswith("-gated"))
    if gated:
        needed.update(GATE_PROXIES)
    result = {} if known is None else {key: value for key, value in known.items() if key in needed}
    for key in [*sorted(needed - set(gated)), *gated]:
        if key in result:
            continue
        mask = expert_gate(result, multiple_only=key in ("una-mc-gated", "option-math-gated")) if key in gated else None
        solver = (
            answer_reader(key.removesuffix("-gated"), published=published)
            if mask is None or mask.any()
            else None
        )
        try:
            result[key] = (
                solver.encode(texts, batch) if mask is None else selective_answers(solver, texts, mask, batch)
            )
        finally:
            if hasattr(solver, "close"):
                solver.close()
        del solver
        gc.collect()
        torch.cuda.empty_cache()
    return result


def question_parts(text, published=False):
    if text.startswith("Q: ") and "\nQ: " in text:
        text = text.rsplit("\nQ: ", 1)[-1]
    text = re.sub(r"\n(?:A|Answer):\s*$", "", text.strip())
    if published:
        text = re.sub(r"(?m)^(\s*)\(([A-D])\)\s*", r"\1\2. ", text)
        if "\n\nQuestion:" in text:
            text = text.rsplit("\n\nQuestion:", 1)[-1]
    markers = list(re.finditer(r"(?:^|\n)[ \t]*([A-D])[.)][ \t]*", text))[-4:]
    if len(markers) != 4 or "".join(m.group(1) for m in markers) != "ABCD":
        return text, []
    stem = text[: markers[0].start()].strip().removesuffix("Choices:").strip()
    options = [
        text[m.end() : markers[i + 1].start() if i < 3 else len(text)].strip() for i, m in enumerate(markers)
    ]
    return stem, options


class AnswerEvidence:
    def __init__(self, proxy, published=False):
        self.published = published
        name, revision = PROXIES[proxy]
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        opts = dict(revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
        if proxy in ("yi", "zephyr"):
            from transformers import BitsAndBytesConfig

            opts.update(
                device_map={"": 0},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
                ),
            )
            self.net = AutoModelForCausalLM.from_pretrained(name, **opts).eval()
        else:
            self.net = AutoModelForCausalLM.from_pretrained(name, **opts).eval().cuda()
        token_ids = [
            [self.tok.encode(prefix + c, add_special_tokens=False)[-1] for c in "ABCD"]
            for prefix in ("", " ")
        ]
        self.letters = torch.as_tensor(token_ids, device="cuda")

    def prompt_ids(self, message):
        ids = self.tok.apply_chat_template([{"role": "user", "content": message}], add_generation_prompt=True)
        return ids if len(ids) <= 384 else ids[:96] + ids[-288:]

    def batch_ids(self, rows):
        ids = self.tok.pad({"input_ids": rows}, padding=True, return_tensors="pt")
        return ids.to("cuda")

    @torch.inference_mode()
    def encode(self, texts, batch):
        evidence = np.zeros((len(texts), 2, 6), dtype=np.float32)
        answers = [""] * len(texts)
        questions = [question_parts(q, self.published) for q in texts]
        for view, order in enumerate(([0, 1, 2, 3], [2, 0, 3, 1])):
            mc = []
            for i, (q, opts) in enumerate(questions):
                if not opts:
                    continue
                options = "\n".join(f"{c}. {opts[j]}" for c, j in zip("ABCD", order))
                message = f"{q}\n{options}\nChoose the correct answer. Reply with only the option letter."
                mc.append((i, self.prompt_ids(message)))
            mc.sort(key=lambda pair: len(pair[1]))
            for start in range(0, len(mc), batch):
                indices, rows = zip(*mc[start : start + batch])
                inputs = self.batch_ids(list(rows))
                hidden = self.net.base_model(**inputs, use_cache=False).last_hidden_state[:, -1:, :]
                logits = self.net.get_output_embeddings()(hidden)[:, 0].float()
                logp = logits.log_softmax(-1)
                # Space-prefixed and unprefixed letters can share a token in some tokenizers.
                letters = torch.logaddexp(logp[:, self.letters[0]], logp[:, self.letters[1]])
                letters -= (self.letters[0] == self.letters[1]).float() * np.log(2)
                probs = letters.softmax(-1)[:, np.argsort(order)]
                extras = torch.stack((letters.logsumexp(-1), logp.max(-1).values), dim=1)
                evidence[list(indices), view] = torch.cat((probs, extras), dim=1).cpu().numpy()
        free = [
            (i, self.prompt_ids(q + "\nGive only the final answer, as briefly as possible."))
            for i, (q, opts) in enumerate(questions)
            if not opts
        ]
        free.sort(key=lambda pair: len(pair[1]))
        for start in range(0, len(free), batch):
            indices, rows = zip(*free[start : start + batch])
            inputs = self.batch_ids(list(rows))
            out = self.net.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=self.tok.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
            ids = out.sequences[:, inputs.input_ids.shape[1] :]
            decoded = self.tok.batch_decode(ids, skip_special_tokens=True)
            conf = torch.stack([s.float().log_softmax(-1).max(-1).values for s in out.scores], dim=1)
            valid = ids != self.tok.pad_token_id
            values = torch.stack((conf[:, 0], (conf * valid).sum(1) / valid.sum(1).clamp_min(1)), dim=1)
            evidence[list(indices), 0, 4:] = values.cpu().numpy()
            for i, ans in zip(indices, decoded):
                answers[i] = ans.strip()
        return evidence, np.asarray(answers, dtype=str)
