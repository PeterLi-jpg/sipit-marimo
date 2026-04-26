"""Fast extraction: pre-compute the candidate hidden-state table ONCE per
prefix, then reuse it across all perturbation cells. Cuts the perturbation
grid from ~30 minutes to ~30 seconds.
"""
from __future__ import annotations

import json
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from transformers.cache_utils import DynamicCache


def expand_pkv(pkv, batch_size):
    if pkv is None:
        return None
    if hasattr(pkv, "layers"):
        out = DynamicCache()
        for li, layer in enumerate(pkv.layers):
            out.update(
                layer.keys.expand(batch_size, -1, -1, -1).contiguous(),
                layer.values.expand(batch_size, -1, -1, -1).contiguous(),
                li,
            )
        return out
    return tuple(
        (k.expand(batch_size, -1, -1, -1).contiguous(),
         v.expand(batch_size, -1, -1, -1).contiguous())
        for k, v in pkv
    )


def candidate_hiddens(model, prefix_ids, layer_idx, batch_size=1024):
    """Forward every vocabulary token through the model with the given prefix
    and return the layer-`layer_idx` hidden state at the last position for each.
    Result is a tensor of shape (vocab_size, hidden_dim).
    """
    vocab_size = model.wte.weight.shape[0]
    pcache = None
    if len(prefix_ids) > 0:
        with torch.no_grad():
            pcache = model(
                torch.tensor(prefix_ids).unsqueeze(0), use_cache=True,
            ).past_key_values

    chunks = []
    for s in range(0, vocab_size, batch_size):
        e = min(s + batch_size, vocab_size)
        bc = torch.arange(s, e).unsqueeze(1)
        with torch.no_grad():
            kw = dict(output_hidden_states=True)
            if pcache is not None:
                kw["past_key_values"] = expand_pkv(pcache, e - s)
            h = model(bc, **kw).hidden_states[layer_idx][:, -1, :]
        chunks.append(h)
    return torch.cat(chunks, dim=0)


def sample_without_true(vocab_size, n, true_id, generator):
    n = max(0, min(n, vocab_size - 1))
    if n == 0:
        return torch.empty((0,), dtype=torch.long)
    draw = torch.randperm(vocab_size - 1, generator=generator)[:n]
    return draw + (draw >= true_id).long()


def landscape_for(model, tokenizer, prompt, layer_idx, n_sample=256, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    true_ids = tokenizer.encode(prompt)
    vocab_size = model.wte.weight.shape[0]

    with torch.no_grad():
        tgt_all = model(
            torch.tensor(true_ids).unsqueeze(0), output_hidden_states=True
        ).hidden_states[layer_idx][0].detach()

    results = []
    for ti, true_id in enumerate(true_ids):
        tgt_h = tgt_all[ti]
        distractors = sample_without_true(vocab_size, n_sample, true_id, g)
        cands = torch.cat([torch.tensor([true_id]), distractors])
        pcache = None
        if ti > 0:
            with torch.no_grad():
                pcache = model(
                    torch.tensor(true_ids[:ti]).unsqueeze(0), use_cache=True,
                ).past_key_values

        losses = []
        batch_size = 512
        for s in range(0, len(cands), batch_size):
            e = min(s + batch_size, len(cands))
            bc = cands[s:e].unsqueeze(1)
            with torch.no_grad():
                kw = dict(output_hidden_states=True)
                if pcache is not None:
                    kw["past_key_values"] = expand_pkv(pcache, e - s)
                h = model(bc, **kw).hidden_states[layer_idx][:, -1, :]
            losses.append(((h - tgt_h) ** 2).mean(1).cpu())

        all_l = torch.cat(losses).numpy()
        tl = float(all_l[0])
        rl = all_l[1:]
        sorted_idx = np.argsort(all_l)[:30]
        results.append({
            "tok_idx": ti,
            "true_token": tokenizer.decode([int(true_id)]),
            "true_loss": tl,
            "rank": int((all_l < tl).sum() + 1),
            "min_rand": float(rl.min()),
            "median_rand": float(np.median(rl)),
            "sample_count": len(rl),
            "top30_losses": [float(x) for x in all_l[sorted_idx]],
            "top30_is_true": [int(i == 0) for i in sorted_idx],
        })
    return results


def recovery_for(model, tokenizer, prompt, layer_idx=12):
    """Use cached candidate-hidden tables to do greedy recovery."""
    true_ids = tokenizer.encode(prompt)
    targets = []
    with torch.no_grad():
        for pos in range(len(true_ids)):
            out = model(
                torch.tensor(true_ids[:pos + 1]).unsqueeze(0),
                output_hidden_states=True,
            )
            targets.append(out.hidden_states[layer_idx][0, -1, :].detach())

    recovered = []
    results = []
    for pos in range(len(true_ids)):
        cand_h = candidate_hiddens(model, recovered, layer_idx)
        diffs = cand_h - targets[pos]
        losses = (diffs ** 2).mean(dim=1)
        best_tok = int(losses.argmin())
        best_loss = float(losses[best_tok])
        recovered.append(best_tok)
        results.append({
            "pos": pos,
            "true_id": true_ids[pos],
            "recovered_id": best_tok,
            "true_word": tokenizer.decode([true_ids[pos]]),
            "recovered_word": tokenizer.decode([best_tok]),
            "min_loss": best_loss,
            "correct": best_tok == true_ids[pos],
        })
    return results


def perturb_grid(model, tokenizer, prompt, noise_levels, quant_levels,
                  layer_idx=12, seed=0):
    """Pre-compute the candidate-hidden table for each prefix ONCE OUTSIDE the
    grid loop, then sweep (noise, quant) by varying only the target side.
    Each grid cell becomes a cheap per-position argmin instead of fresh
    50K-vocab forward sweeps."""
    true_ids = tokenizer.encode(prompt)
    clean = []
    with torch.no_grad():
        for pos in range(len(true_ids)):
            out = model(
                torch.tensor(true_ids[:pos + 1]).unsqueeze(0),
                output_hidden_states=True,
            )
            clean.append(out.hidden_states[layer_idx][0, -1, :].detach())

    # Cache candidate-hiddens per prefix tuple. The hot path: small noise →
    # recovery succeeds at every position → prefix at pos i is true_ids[:i].
    # We pre-compute those once. Bigger noise may flip tokens and need new
    # prefixes; we lazily compute and memoize those too.
    cand_cache = {}
    def get_cand(prefix):
        key = tuple(prefix)
        if key not in cand_cache:
            print(f"    [computing candidate table for prefix {key}]",
                  file=sys.stderr)
            cand_cache[key] = candidate_hiddens(model, list(prefix), layer_idx)
        return cand_cache[key]

    # Warm the cache for the expected clean prefixes
    for pos in range(len(true_ids)):
        get_cand(true_ids[:pos])

    def add_noise(t, r, sd):
        if r <= 0.0:
            return t.clone()
        gg = torch.Generator()
        gg.manual_seed(sd)
        n = torch.randn(t.shape, generator=gg)
        return t + r * n / (n.norm() + 1e-12)

    def quantize(t, bits):
        if bits <= 0:
            return t.clone()
        mx = t.abs().max()
        if mx == 0:
            return t.clone()
        levels = (2 ** bits) - 1
        scaled = (t / mx + 1.0) / 2.0
        return ((scaled * levels).round() / levels * 2.0 - 1.0) * mx

    out = {}
    for nr in noise_levels:
        for qb in quant_levels:
            print(f"  perturb cell noise={nr} quant={qb}", file=sys.stderr)
            perturbed = []
            pert_norms = []
            for pos, tgt in enumerate(clean):
                noisy = add_noise(tgt, nr, seed * 1000 + pos)
                quant = quantize(noisy, qb)
                perturbed.append(quant)
                pert_norms.append(float((quant - tgt).norm()))

            recovered = []
            results = []
            for pos in range(len(true_ids)):
                cand_h = get_cand(recovered)  # cached!
                diffs = cand_h - perturbed[pos]
                losses = (diffs ** 2).mean(dim=1)
                best_tok = int(losses.argmin())
                best_loss = float(losses[best_tok])
                recovered.append(best_tok)
                results.append({
                    "pos": pos,
                    "true_word": tokenizer.decode([true_ids[pos]]),
                    "recovered_word": tokenizer.decode([best_tok]),
                    "perturbation_norm": pert_norms[pos],
                    "min_loss": best_loss,
                    "correct": best_tok == true_ids[pos],
                })
            out[(nr, qb)] = results
    return out


def main():
    print("loading GPT-2 small …", file=sys.stderr)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.ln_f = nn.Identity()
    model.eval()

    # ── PCA basis (recompute from base sentences, then project demos) ─────────
    BASE_SENTENCES = [
        "The cat sat on the mat",
        "A cat sat on a mat",
        "The dog ran through the park",
        "Machine learning is transforming science",
        "Paris is the capital of France",
        "To be or not to be",
        "The quick brown fox jumps",
        "Statistical inference and probability",
    ]
    print("PCA basis …", file=sys.stderr)
    base_h = []
    with torch.no_grad():
        for s in BASE_SENTENCES:
            ids = tokenizer.encode(s, return_tensors="pt")
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
            base_h.append(h)
    base_H = np.stack(base_h, axis=0)
    mean = base_H.mean(axis=0)
    centered = base_H - mean
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:2]
    explained_var = (s ** 2) / (centered.shape[0] - 1)
    explained_ratio = (explained_var[:2] / explained_var.sum()).tolist()
    XY_BASE = (base_H - mean) @ components.T

    DEMO_SENTENCES = [
        "Hello world",
        "Hello world how",
        "The standard model of particle physics",
        "Curiosity drove the rover across Mars",
        "I think therefore I am",
        "Time flies like an arrow",
        "The brown fox is quick and clever",
        "Bach composed the Brandenburg concertos",
        "All happy families are alike",
        "The hidden state encodes everything",
    ]
    print("demo PCA projections …", file=sys.stderr)
    demo_xy = {}
    with torch.no_grad():
        for s in DEMO_SENTENCES:
            ids = tokenizer.encode(s, return_tensors="pt")
            out = model(ids, output_hidden_states=True)
            h = out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
            demo_xy[s] = ((h - mean) @ components.T).tolist()

    min_pair = float("inf")
    for i in range(len(XY_BASE)):
        for j in range(i + 1, len(XY_BASE)):
            d = float(np.linalg.norm(XY_BASE[i] - XY_BASE[j]))
            min_pair = min(min_pair, d)

    # ── Landscapes (4 configurations) ─────────────────────────────────────────
    LANDSCAPE_CONFIGS = [
        ("The cat sat on the mat", 12),
        ("Hello world how", 12),
        ("Paris is the capital", 12),
        ("The cat sat on the mat", 8),
    ]
    landscape_blob = {}
    for prompt, layer_idx in LANDSCAPE_CONFIGS:
        print(f"landscape '{prompt}' @ layer {layer_idx} …", file=sys.stderr)
        landscape_blob[f"{prompt}|{layer_idx}"] = landscape_for(
            model, tokenizer, prompt, layer_idx,
        )

    # ── Recoveries (3 prompts) ────────────────────────────────────────────────
    RECOVERY_PROMPTS = ["Hello world how", "The cat sat", "Quantum"]
    recovery_blob = {}
    for prompt in RECOVERY_PROMPTS:
        print(f"recovery '{prompt}' …", file=sys.stderr)
        recovery_blob[prompt] = recovery_for(model, tokenizer, prompt)

    # ── Perturbation grid (5 × 3) ─────────────────────────────────────────────
    NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 5.0]
    QUANT_LEVELS = [0, 8, 4]
    print("perturb grid (cached candidates → fast) …", file=sys.stderr)
    p_blob = perturb_grid(
        model, tokenizer, "Hello world", NOISE_LEVELS, QUANT_LEVELS,
    )
    perturb_serial = {f"{k[0]}|{k[1]}": v for k, v in p_blob.items()}

    # ── Steganography channel data ────────────────────────────────────────────
    # For § 6 of the notebook we need, for the cover prompt "Hello world":
    #   - the clean layer-12 hidden state at every position (the "carrier")
    #   - the TRUE half-margin (min distance from the true-token hidden state
    #     to ANY other vocabulary token's hidden state, evaluated at the
    #     correct prefix). This is the per-position noise budget under
    #     Theorem 3.2: any perturbation with ||δ|| < margin/2 is decoded back
    #     to the original token.
    print("steganography data: clean carrier + true half-margins …", file=sys.stderr)
    STEGO_PROMPT = "Hello world"
    stego_ids = tokenizer.encode(STEGO_PROMPT)
    stego_clean = []
    stego_margins = []
    layer_idx = 12
    with torch.no_grad():
        for pos in range(len(stego_ids)):
            out = model(
                torch.tensor(stego_ids[:pos + 1]).unsqueeze(0),
                output_hidden_states=True,
            )
            true_h = out.hidden_states[layer_idx][0, -1, :].detach()
            stego_clean.append(true_h.cpu().numpy().tolist())

            # Distance from true_h to every other vocab token's hidden state
            # at the same prefix → minimum is the per-position margin.
            cand_h = candidate_hiddens(model, stego_ids[:pos], layer_idx)
            diffs = cand_h - true_h.unsqueeze(0)
            dists = diffs.norm(dim=1)
            # Exclude the true token itself (its distance is zero)
            true_id = stego_ids[pos]
            dists[true_id] = float("inf")
            stego_margins.append(float(dists.min()))

    out = {
        "BASE_SENTENCES": BASE_SENTENCES,
        "XY_BASE": XY_BASE.tolist(),
        "PCA_COMPONENTS": components.tolist(),
        "PCA_MEAN": mean.tolist(),
        "PCA_EXPLAINED_VARIANCE_RATIO": explained_ratio,
        "MIN_PAIRWISE_DIST": min_pair,
        "DEMO_SENTENCES": DEMO_SENTENCES,
        "DEMO_XY": demo_xy,
        "LANDSCAPES": landscape_blob,
        "RECOVERIES": recovery_blob,
        "PERTURB_PROMPT": "Hello world",
        "PERTURB_NOISE_LEVELS": NOISE_LEVELS,
        "PERTURB_QUANT_LEVELS": QUANT_LEVELS,
        "PERTURB_RESULTS": perturb_serial,
        "STEGO_PROMPT": STEGO_PROMPT,
        "STEGO_TOKEN_IDS": stego_ids,
        "STEGO_TOKENS": [tokenizer.decode([t]) for t in stego_ids],
        "STEGO_CLEAN_HIDDENS": stego_clean,
        "STEGO_HALF_MARGINS": stego_margins,
    }
    with open("toolkit_data.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote toolkit_data.json", file=sys.stderr)


if __name__ == "__main__":
    main()
