"""Text-conditioned prediction of the historical models' response probabilities."""

import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


BACKBONES = {
    "small": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "large": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "expert": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
}


def tokenizer_for(size):
    name, rev = BACKBONES[size]
    return AutoTokenizer.from_pretrained(name, revision=rev)


def tokenize_prompts(tokenizer, texts, limit):
    """Keep the question and its answer slot when a long multiple-choice prompt is truncated."""
    texts = [re.split(r"(?:^|\n)Q: ", t)[-1] if t.startswith("Q: ") and "\nQ: " in t else t for t in texts]
    ids = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    tokens = np.full((len(ids), limit), tokenizer.pad_token_id, dtype=np.int32)
    lengths = np.empty(len(ids), dtype=np.int32)
    for i, row in enumerate(ids):
        if len(row) > limit:
            keep = limit // 3
            row = row[:keep] + row[-(limit - keep) :]
        if not row:
            raise ValueError("A routing prompt must contain text.")
        lengths[i] = len(row)
        tokens[i, : len(row)] = row
    return torch.from_numpy(tokens), torch.from_numpy(lengths)


class ResponseEncoder(nn.Module):
    def __init__(self, size, n_models, n_domains, tune_blocks):
        super().__init__()
        name, rev = BACKBONES[size]
        self.backbone = AutoModel.from_pretrained(
            name, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.backbone.requires_grad_(False)
        for block in self.backbone.layers[-tune_blocks:]:
            block.float().requires_grad_(True)
        hidden = self.backbone.config.hidden_size
        tok = tokenizer_for(size)
        ids = [
            tok.encode(s, add_special_tokens=False)[-1] for s in (" A", " B", " C", " D", "A", "B", "C", "D")
        ]
        self.register_buffer("answer_tokens", torch.tensor(ids), persistent=False)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2 + 8, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(256, n_models + n_domains),
        )
        self.n_models = n_models

    def forward(self, tokens, lengths):
        rows = torch.arange(len(tokens), device=tokens.device)
        valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.backbone(tokens, attention_mask=valid, use_cache=False).last_hidden_state
            last = hidden[rows, lengths.long() - 1]
            average = (hidden * valid[:, :, None]).sum(1) / lengths[:, None]
            letters = self.backbone.embed_tokens.weight[self.answer_tokens]
            option_logits = F.linear(last, letters).float().reshape(-1, 2, 4)
            options = (option_logits - option_logits.mean(dim=2, keepdim=True)).flatten(1)
            features = torch.cat(
                [F.layer_norm(last, (last.shape[-1],)), F.layer_norm(average, (average.shape[-1],)), options],
                dim=1,
            )
            out = self.readout(features).float()
        return out[:, : self.n_models], out[:, self.n_models :]

    @torch.inference_mode()
    def predict(self, tokens, lengths, batch):
        self.eval()
        order = torch.argsort(lengths)
        replies, memberships = [], []
        for group in order.split(batch):
            width = int(lengths[group].max())
            logits, domain_logits = self(tokens[group, :width].cuda(), lengths[group].cuda())
            replies.append(logits.sigmoid().cpu().numpy())
            memberships.append(domain_logits.softmax(dim=1).cpu().numpy())
        undo = order.argsort().cpu().numpy()
        return np.concatenate(replies)[undo], np.concatenate(memberships)[undo]

    def save_tuned(self, path, spec):
        trainable = {
            name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad
        }
        torch.save({"spec": spec, "parameters": trainable}, path)

    @classmethod
    def restore(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        spec = payload["spec"]
        net = cls(spec["size"], spec["n_models"], spec["n_domains"], spec["tune_blocks"])
        net.load_state_dict(payload["parameters"], strict=False)
        return net.cuda().eval(), spec
