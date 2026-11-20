"""Represent a question by a frozen language model's estimates of its skill requirements."""

import numpy as np
import torch

from .answer_evidence import AnswerEvidence, question_parts


RUBRIC = (
    (
        "Which background is most important?",
        (
            "Medicine or biology",
            "Mathematics, computing or formal logic",
            "Social science, humanities or law",
            "Other natural science or everyday knowledge",
        ),
    ),
    (
        "What does answering require most?",
        (
            "Retrieving a specific fact",
            "Calculating a quantity",
            "Deducing a conclusion from constraints",
            "Interpreting intentions, meaning or context",
        ),
    ),
    (
        "What makes this question difficult?",
        (
            "An obscure specialized fact",
            "Several reasoning steps",
            "A subtle distinction between plausible answers",
            "It is straightforward",
        ),
    ),
    (
        "Which training would help a language model most on this item?",
        (
            "Biomedical texts and clinical cases",
            "Code, algorithms and formal logic",
            "Mathematical problem solving",
            "Broad factual knowledge and instruction following",
        ),
    ),
    (
        "What medical knowledge is needed?",
        (
            "Clinical diagnosis or treatment",
            "Anatomy, pathology or basic biomedical science",
            "Population health or quantitative evidence",
            "No substantive medical expertise",
        ),
    ),
    (
        "Which quantitative operation matters most?",
        (
            "Short arithmetic or unit conversion",
            "Setting up an equation with multiple steps",
            "Symbolic, combinatorial or algorithmic reasoning",
            "No substantive quantitative operation",
        ),
    ),
    (
        "How much information must be combined?",
        (
            "One learned association",
            "Two simple facts or conditions",
            "A chain of several constraints",
            "The main challenge is an ambiguous or misleading premise",
        ),
    ),
    (
        "Which mistake is most likely for a weaker model?",
        (
            "Guessing a fact it does not know",
            "Making a numerical or procedural mistake",
            "Reversing an exclusion or logical relation",
            "Following a misleading surface cue",
        ),
    ),
)


class SkillAxes(AnswerEvidence):
    @torch.inference_mode()
    def encode_axes(self, texts, batch=32):
        out = np.empty((len(texts), len(RUBRIC), 4), np.float32)
        clean = []
        for text in texts:
            stem, opts = question_parts(text)
            clean.append(stem + ("\n" + "\n".join(opts) if opts else ""))
        for axis, (question, choices) in enumerate(RUBRIC):
            order = np.roll(np.arange(4), axis % 4)
            scale = "\n".join(f"{letter}. {choices[i]}" for letter, i in zip("ABCD", order))
            rows = [
                self.prompt_ids(
                    f"Classify the following problem.\n<problem>\n{q}\n</problem>\n{question}\n{scale}\nReply with the option letter."
                )
                for q in clean
            ]
            ranks = np.argsort([len(r) for r in rows])
            for i in range(0, len(rows), batch):
                ix = ranks[i : i + batch]
                inputs = self.batch_ids([rows[j] for j in ix])
                hidden = self.net.base_model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
                weight = self.net.get_output_embeddings().weight[self.letters.flatten()]
                logits = (hidden @ weight.T).float().reshape(-1, 2, 4)
                scores = torch.logaddexp(logits[:, 0], logits[:, 1]) - (
                    self.letters[0] == self.letters[1]
                ).float() * np.log(2)
                out[ix, axis] = scores.softmax(-1)[:, np.argsort(order)].cpu().numpy()
        return out
