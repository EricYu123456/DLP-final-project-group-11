"""Read the original 70B AWQ candidate one decoder layer at a time on the 200 onboarding prompts."""

from contextlib import ExitStack
import json

import numpy as np
from safetensors import safe_open
import torch
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

from coldstart.answer_evidence import question_parts


def unpack_gemm(get, prefix, group_size):
    # The published AutoAWQ GEMM reverse order is 0,4,1,5,2,6,3,7 within a packed int32.
    # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py
    codes, zeros, scales = [get(prefix + suffix) for suffix in (".qweight", ".qzeros", ".scales")]
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=codes.device, dtype=torch.int32)

    def unpack(t):
        return ((t[..., None] >> shifts) & 15).reshape(t.shape[0], -1).to(scales.dtype)

    zero = unpack(zeros)
    weight = torch.empty((scales.shape[1], codes.shape[0]), dtype=scales.dtype, device=codes.device)
    for i in range(0, len(codes), 2 * group_size):
        block = unpack(codes[i : i + 2 * group_size])
        groups = torch.arange(i, i + len(block), device=codes.device) // group_size
        weight[:, i : i + len(block)] = ((block - zero[groups]) * scales[groups]).T
    return weight


@torch.inference_mode()
def one_token_panel(path, texts):
    cfg = LlamaConfig.from_pretrained(path, local_files_only=True)
    cfg._attn_implementation = "sdpa"
    group_size = cfg.quantization_config["group_size"]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = tok(list(map(str, texts)), add_special_tokens=True)["input_ids"]
    lengths = np.array(list(map(len, ids)))
    ids = [v[:96] + v[-894:] if len(v) > 990 else v for v in ids]
    ordered = np.argsort([len(v) for v in ids], kind="stable")
    groups = [ordered[i : i + 6] for i in range(0, len(ordered), 6)]
    mapping = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    with ExitStack() as files:
        readers = {
            name: files.enter_context(safe_open(str(path / name), framework="pt", device="cpu"))
            for name in set(mapping.values())
        }

        def get(key):
            return readers[mapping[key]].get_tensor(key).cuda()

        embedding = get("model.embed_tokens.weight")
        states = []
        for batch in groups:
            limit = max(len(ids[i]) for i in batch)
            padded = [ids[i] + [0] * (limit - len(ids[i])) for i in batch]
            states.append(embedding[torch.tensor(padded, device="cuda")])
        del embedding
        limit = max(q.shape[1] for q in states)
        pos = torch.arange(limit, device="cuda")[None]
        rotary = LlamaRotaryEmbedding(cfg, device="cuda")
        cos, sin = rotary(states[0], pos)
        for i in range(cfg.num_hidden_layers):
            with torch.device("meta"):
                layer = LlamaDecoderLayer(cfg, i).eval()
            weights = {}
            for name in layer.state_dict():
                key = f"model.layers.{i}.{name}"
                weights[name] = (
                    get(key) if key in mapping else unpack_gemm(get, key.removesuffix(".weight"), group_size)
                )
            layer.load_state_dict(weights, assign=True)
            for j, h in enumerate(states):
                length = h.shape[1]
                # All padding follows the original text. A causal prefix never attends to that padding.
                states[j] = layer(
                    h, position_ids=pos[:, :length], position_embeddings=(cos[:, :length], sin[:, :length])
                )
            del layer, weights, h
            if (i + 1) % 10 == 0:
                print(f"AWQ {i + 1}/{cfg.num_hidden_layers} original decoder layers", flush=True)
        norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        norm.load_state_dict({"weight": get("model.norm.weight")}, assign=True)
        head = get("lm_head.weight")
        logits = np.empty((len(texts), head.shape[0]), np.float32)
        for batch, h in zip(groups, states):
            positions = torch.tensor([len(ids[j]) - 1 for j in batch], device="cuda")
            last = norm(h[torch.arange(len(batch), device="cuda"), positions])
            logits[batch] = (last @ head.T).float().cpu().numpy()
    from scipy.special import logsumexp, softmax

    letters = {
        letter: {tok.encode(s, add_special_tokens=False)[-1] for s in (letter, " " + letter)}
        for letter in "ABCD"
    }
    multiple = np.array([bool(question_parts(t)[1]) for t in texts])
    ln = logits - logsumexp(logits, axis=1, keepdims=True)
    option = np.stack([logsumexp(ln[:, sorted(indices)], axis=1) for indices in letters.values()], axis=1)
    prob = np.zeros((len(texts), 6), np.float32)
    prob[multiple, :4] = softmax(option[multiple], axis=1)
    prob[multiple, 4] = logsumexp(option[multiple], axis=1)
    prob[multiple, 5] = option[multiple].max(1)
    top = np.argmax(logits, axis=1)
    prob[~multiple, 4:] = ln[np.arange(len(top)), top][~multiple, None]
    leading = np.array([tok.decode([int(v)]) for v in top])
    # A single unconstrained token does not identify a complete free response for this likelihood-only arm.
    answers = np.full(len(texts), "", dtype="U1")
    return (
        prob,
        answers,
        dict(
            leading_token=leading,
            free_completion_tokens=1,
            truncated_questions=int((lengths > 990).sum()),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        ),
    )
