"""Question-only language-model features for estimating historical LLM strengths."""

import re

import numpy as np
import torch
from transformers import AutoModel

from .capability_text import BACKBONES, tokenize_prompts, tokenizer_for


def question_views(text):
    text = text.strip()
    if text.startswith("Q: ") and "\nQ: " in text:
        text = text.rsplit("\nQ: ", 1)[-1]
    stem = re.sub(r"\n(?:A|Answer):\s*$", "", text)
    markers = list(re.finditer(r"(?:^|\n)[ \t]*([A-D])\.[ \t]*", stem))[-4:]
    if len(markers) == 4 and "".join(m.group(1) for m in markers) == "ABCD":
        choices = [
            stem[m.end() : markers[i + 1].start() if i < 3 else len(stem)].strip()
            for i, m in enumerate(markers)
        ]
        prefix = stem[: markers[0].start()].removesuffix("Choices:").rstrip()
        rotated = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip("ABCD", choices[2:] + choices[:2])
        )
        alternate = f"{prefix}\n{rotated}\nAnswer:"
    else:
        alternate = (
            f"{stem}\nWhich skill matters most here?\nA. Factual recall\nB. Calculation\n"
            "C. Logical inference\nD. Recognizing a false premise\nAnswer:"
        )
    difficulty = (
        f"{stem}\nHow difficult would this question be for a small language model?\n"
        "A. Very easy\nB. Moderately easy\nC. Moderately difficult\nD. Very difficult\nAnswer:"
    )
    return text, alternate, difficulty


class FrozenQuestionViews:
    def __init__(self, size):
        name, revision = BACKBONES[size]
        self.tok = tokenizer_for(size)
        self.net = (
            AutoModel.from_pretrained(
                name, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        alternatives = [" A", " B", " C", " D", "A", "B", "C", "D", " yes", " no", " Yes", " No"]
        indices = [self.tok.encode(a, add_special_tokens=False)[-1] for a in alternatives]
        self.readout = self.net.embed_tokens.weight[indices].detach()

    @torch.inference_mode()
    def encode(self, texts, batch, limit=256):
        views = list(zip(*(question_views(t) for t in texts)))
        parts = []
        for j, view in enumerate(views):
            ids, lens = tokenize_prompts(self.tok, list(view), limit)
            order = lens.argsort()
            features = []
            for ix in order.split(batch):
                lengths = lens[ix].cuda()
                tokens = ids[ix, : int(lens[ix].max())].cuda()
                valid = torch.arange(tokens.shape[1], device="cuda")[None] < lengths[:, None]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = self.net(tokens, attention_mask=valid, use_cache=False).last_hidden_state
                    last = hidden[torch.arange(len(ix), device="cuda"), lengths.long() - 1]
                    logits = (last @ self.readout.T).float()
                # Keep pretrained semantic features from the unmodified question and answer evidence from all views.
                fields = [logits, logits[:, :4].softmax(1), logits[:, 4:8].softmax(1)]
                if j == 0:
                    average = (hidden.float() * valid[:, :, None]).sum(1) / lengths[:, None]
                    fields = [last.float(), average, *fields]
                features.append(torch.cat(fields, dim=1).cpu().numpy())
            parts.append(np.concatenate(features)[order.argsort().numpy()])
        return np.column_stack(parts).astype(np.float32)
