"""Score answer texts with conditional likelihood and a neutral-context likelihood correction."""

import numpy as np
from scipy.special import softmax
import torch

from .answer_evidence import AnswerEvidence, question_parts


class OptionLikelihood:
    def __init__(self, published=False):
        self.reader = AnswerEvidence("math", published)
        self.neutral = self.tokens(
            "Question: Choose the most plausible answer.\nAnswer:"
        )
        self.prior = {}

    def tokens(self, text):
        return self.reader.tok.encode(text, add_special_tokens=False)

    @torch.inference_mode()
    def log_likelihoods(self, entries, batch):
        values = np.zeros(len(entries), np.float32)
        order = sorted(range(len(entries)), key=lambda i: sum(map(len, entries[i])))
        for start in range(0, len(order), batch):
            indices = order[start : start + batch]
            examples = [entries[i] for i in indices]
            lengths = [len(p) + len(a) for p, a in examples]
            rows = [
                list(p) + list(a) + [self.reader.tok.pad_token_id] * (max(lengths) - n)
                for (p, a), n in zip(examples, lengths)
            ]
            input_ids = torch.tensor(rows, device="cuda")
            mask = (
                torch.arange(input_ids.shape[1], device="cuda")[None]
                < torch.tensor(lengths, device="cuda")[:, None]
            )
            hidden = self.reader.net.base_model(
                input_ids=input_ids, attention_mask=mask, use_cache=False
            ).last_hidden_state
            selected, target = [], []
            for j, (prefix, answer) in enumerate(examples):
                selected.append(
                    hidden[j, len(prefix) - 1 : len(prefix) + len(answer) - 1]
                )
                target.extend(answer)
            chosen = torch.cat(selected)
            target = torch.tensor(target, device="cuda")
            # Only materialize vocabulary logits for answer tokens, in bounded groups on the 16 GB card.
            logp = []
            for begin in range(0, len(chosen), 64):
                z = self.reader.net.get_output_embeddings()(
                    chosen[begin : begin + 64]
                ).float()
                words = target[begin : begin + len(z)]
                logp.append(z.gather(1, words[:, None]).squeeze(1) - z.logsumexp(1))
            token_scores = torch.cat(logp).cpu().numpy()
            offset = 0
            for i, (_, answer) in zip(indices, examples):
                values[i] = token_scores[offset : offset + len(answer)].mean()
                offset += len(answer)
        return values

    def encode(self, texts, batch=16):
        requests, locations = [], []
        for i, text in enumerate(texts):
            stem, options = question_parts(text, self.reader.published)
            if not options:
                continue
            prefix = self.tokens(f"Question: {stem}\nAnswer:")
            if len(prefix) > 192:
                prefix = prefix[:48] + prefix[-144:]
            for j, option in enumerate(options):
                ids = self.tokens(" " + option.strip()) or [
                    self.reader.tok.eos_token_id
                ]
                answer = tuple(ids if len(ids) <= 48 else ids[:24] + ids[-24:])
                requests.append((prefix, answer))
                locations.append((i, j))
        missing = list(dict.fromkeys(a for _, a in requests if a not in self.prior))
        if missing:
            self.prior.update(
                zip(
                    missing,
                    self.log_likelihoods([(self.neutral, a) for a in missing], batch),
                )
            )
        scores = self.log_likelihoods(requests, batch)
        raw = np.zeros((len(texts), 2, 4), np.float32)
        out = np.zeros((len(texts), 2, 6), np.float32)
        for (i, j), (_, a), s in zip(locations, requests, scores):
            raw[i, :, j] = s, s - self.prior[a]
        if locations:
            valid = np.unique([i for i, _ in locations])
            out[valid, :, :4] = softmax(raw[valid], axis=-1)
            out[valid, :, 4] = raw[valid].max(-1)
            out[valid, :, 5] = raw[valid].mean(-1)
        return out, np.full(len(texts), "", dtype="U1")
