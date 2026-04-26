# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.23.0",
#   "torch>=2.3",
#   "transformers>=4.50.0",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "scikit-learn>=1.4",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(
    width="medium",
    app_title="LMs Are Injective: GPT-2 + CPU Robustness Demo",
)


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from transformers import GPT2Model, GPT2Tokenizer
    from transformers.cache_utils import DynamicCache

    return DynamicCache, GPT2Model, GPT2Tokenizer, PCA, mo, nn, np, plt, torch


@app.cell
def _(DynamicCache, torch):
    def expand_past_key_values(past_key_values, batch_size):
        if past_key_values is None:
            return None

        if hasattr(past_key_values, "layers"):
            expanded = DynamicCache()
            for layer_idx, layer in enumerate(past_key_values.layers):
                expanded.update(
                    layer.keys.expand(batch_size, -1, -1, -1).contiguous(),
                    layer.values.expand(batch_size, -1, -1, -1).contiguous(),
                    layer_idx,
                )
            return expanded

        expanded = []
        for key, value in past_key_values:
            expanded.append(
                (
                    key.expand(batch_size, -1, -1, -1).contiguous(),
                    value.expand(batch_size, -1, -1, -1).contiguous(),
                )
            )
        return tuple(expanded)

    def sample_without_true(vocab_size, sample_size, true_id, device):
        sample_size = max(0, min(sample_size, vocab_size - 1))
        if sample_size == 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        draw = torch.randperm(vocab_size - 1, device=device)[:sample_size]
        return draw + (draw >= true_id).long()

    def add_noise_to_tensor(tensor, radius, seed):
        """Add fixed-radius isotropic noise to a 1-D hidden-state tensor."""
        if radius <= 0.0:
            return tensor.clone()
        generator = torch.Generator()
        generator.manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator)
        noise = noise / (noise.norm() + 1e-12)
        return tensor + radius * noise

    def quantize_tensor(tensor, bits):
        """Uniform min-max quantization of a hidden-state tensor."""
        if bits <= 0:
            return tensor.clone()
        max_abs = tensor.abs().max()
        if max_abs == 0:
            return tensor.clone()
        levels = (2 ** bits) - 1
        scaled = (tensor / max_abs + 1.0) / 2.0
        quantized = (scaled * levels).round() / levels
        return (quantized * 2.0 - 1.0) * max_abs

    return add_noise_to_tensor, expand_past_key_values, quantize_tensor, sample_without_true


@app.cell
def _(GPT2Model, GPT2Tokenizer, mo, nn, torch):
    with mo.status.spinner(title="Loading GPT-2 on CPU (~500 MB on first run)..."):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2Model.from_pretrained("gpt2")

        # This makes the loss landscape numerically cleaner for a demo notebook.
        # It is a notebook choice, not a theorem requirement.
        model.ln_f = nn.Identity()
        model.eval()
        device = torch.device("cpu")
        model = model.to(device)

    return device, model, tokenizer


@app.cell
def _(mo):
    mo.md(
        r"""
        # Language Models Are Injective and Hence Invertible

        **Paper:** [Language Models are Injective and Hence Invertible](https://arxiv.org/abs/2510.15511)
        by Nikolaou et al. (ICLR 2026)

        This notebook focuses on the paper's core security claim in the **hidden-state leakage**
        setting: if an adversary obtains the per-position hidden states of a decoder-only language
        model at a fixed layer, the original prompt can be reconstructed token by token.

        Two things matter for this notebook:

        - The paper's theorems are about **hidden representations**, not recovering prompts from
          final output text alone.
        - The proof is mathematical, but the notebook should still be **CPU-friendly** and
          interactive.

        So this notebook does two jobs:

        - it shows a real GPT-2 hidden-state inversion demo on CPU,
        - it adds a CPU-cheap extension for **Theorem 3.2 style noise robustness**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## § 1  Injectivity in Practice: A Visual Sanity Check

        The paper's formal guarantees are Theorem **2.2** (almost-sure injectivity at initialization)
        and Theorem **2.3** (injectivity preserved under finite gradient training).

        The plot below is **not a proof** of injectivity. It is just an illustration that nearby or
        semantically similar prompts still land at distinct GPT-2 hidden representations. The paper's
        stronger evidence comes from billions of collision checks across large prompt sets.
        """
    )
    return


@app.cell
def _(PCA, device, model, np, tokenizer, torch):
    # Compute base hidden states once at load time; PCA is fit here and reused
    # when the user adds a custom sentence below.
    base_sentences = [
        "The cat sat on the mat",
        "A cat sat on a mat",
        "The dog ran through the park",
        "Machine learning is transforming science",
        "Paris is the capital of France",
        "To be or not to be",
        "The quick brown fox jumps",
        "Statistical inference and probability",
    ]

    _vectors = []
    with torch.no_grad():
        for _s in base_sentences:
            _ids = tokenizer.encode(_s, return_tensors="pt").to(device)
            _out = model(_ids, output_hidden_states=True)
            _h = _out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
            _vectors.append(_h)

    _vectors = np.array(_vectors)
    pca = PCA(n_components=2)
    xy_base = pca.fit_transform(_vectors)

    min_pairwise_dist = min(
        np.linalg.norm(_vectors[i] - _vectors[j])
        for i in range(len(_vectors))
        for j in range(i + 1, len(_vectors))
    )

    return base_sentences, min_pairwise_dist, pca, xy_base


@app.cell
def _(mo):
    sentence_input = mo.ui.text(
        value="",
        placeholder="Type a sentence and press Enter...",
        label="Add your own sentence to the plot",
        full_width=True,
    )
    sentence_input
    return (sentence_input,)


@app.cell
def _(
    base_sentences,
    device,
    min_pairwise_dist,
    mo,
    model,
    np,
    pca,
    plt,
    sentence_input,
    tokenizer,
    torch,
    xy_base,
):
    custom = (sentence_input.value or "").strip()

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_sentences)))

    for idx, (point, sentence) in enumerate(zip(xy_base, base_sentences)):
        ax.scatter(*point, s=160, color=colors[idx], zorder=5)
        ax.annotate(
            f'"{sentence}"',
            point,
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=colors[idx],
        )

    if custom:
        # Project the user's sentence into the existing PCA space (no refit)
        with torch.no_grad():
            _ids = tokenizer.encode(custom, return_tensors="pt").to(device)
            _out = model(_ids, output_hidden_states=True)
            _h = _out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
        xy_custom = pca.transform(_h.reshape(1, -1))[0]
        ax.scatter(*xy_custom, s=280, color="black", marker="*", zorder=10)
        ax.annotate(
            f'"{custom}"',
            xy_custom,
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
            fontweight="bold",
        )

    n_pairs = len(base_sentences) * (len(base_sentences) - 1) // 2
    ax.set_title(
        "GPT-2 last-layer hidden states — PCA projection (mean pooled)\n"
        f"Minimum pairwise L2 distance: {min_pairwise_dist:.1f}",
        fontsize=10.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance explained)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance explained)")
    ax.grid(True, alpha=0.2)

    mo.vstack(
        [
            fig,
            mo.md(
                f"Minimum pairwise L2 distance across {n_pairs} prompt pairs: "
                f"**{min_pairwise_dist:.2f}**. Each sentence lands at a distinct point — "
                "consistent with injectivity, though not a proof of it."
            ).callout(kind="neutral"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## § 2  Why Inversion Works: One-Step Loss Landscapes

        At each position, once the prefix is known, the paper studies the one-step map
        $v \mapsto h_t(\pi \oplus v)$. If that map is injective, only one candidate token matches the
        observed hidden state.

        The visualization below samples a set of distractor tokens and compares their hidden states
        against the target. In exact arithmetic, the true token is the unique minimizer; in practice
        you should expect a **near-zero** loss for the true token and clearly larger losses for
        sampled impostors.
        """
    )
    return


@app.cell
def _(mo):
    prompt_input = mo.ui.text(
        value="The cat sat on the mat",
        label="Prompt",
        full_width=True,
    )
    layer_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=12,
        label="Target layer",
        show_value=True,
        full_width=True,
    )
    n_sample_slider = mo.ui.slider(
        start=128,
        stop=2048,
        step=128,
        value=512,
        label="Unique distractor tokens per position",
        show_value=True,
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="▶ Inspect Loss Landscape")

    mo.vstack(
        [
            mo.md("### Controls"),
            prompt_input,
            mo.hstack([layer_slider, n_sample_slider], widths="equal"),
            run_btn,
        ],
        gap=0.6,
    )
    return layer_slider, n_sample_slider, prompt_input, run_btn


@app.cell
def _(mo, prompt_input, tokenizer):
    ids = tokenizer.encode(prompt_input.value or "")
    preview = "  ".join(f"`{tokenizer.decode([token])}`" for token in ids) if ids else "_empty_"
    mo.md(f"**Tokens ({len(ids)}):** {preview}")
    return


@app.cell
def _(
    device,
    expand_past_key_values,
    layer_slider,
    model,
    mo,
    n_sample_slider,
    np,
    prompt_input,
    run_btn,
    sample_without_true,
    tokenizer,
    torch,
):
    if not run_btn.value:
        landscape_prompt = None
        landscape_results = None
        return landscape_prompt, landscape_results

    prompt = (prompt_input.value or "").strip()
    layer_idx = int(layer_slider.value)
    sample_count = int(n_sample_slider.value)

    true_ids_list = tokenizer.encode(prompt)
    if not true_ids_list:
        landscape_prompt = prompt
        landscape_results = []
        return landscape_prompt, landscape_results

    true_ids = torch.tensor(true_ids_list).unsqueeze(0).to(device)
    vocab_size = model.wte.weight.shape[0]
    batch_size = 512

    with torch.no_grad():
        target_output = model(true_ids, output_hidden_states=True)
    target_all = target_output.hidden_states[layer_idx][0].detach()

    results = []
    with mo.status.progress_bar(total=len(true_ids_list), title="Computing loss landscapes...") as bar:
        for token_idx in range(len(true_ids_list)):
            target_hidden = target_all[token_idx]
            true_id = true_ids_list[token_idx]

            distractors = sample_without_true(
                vocab_size=vocab_size,
                sample_size=sample_count,
                true_id=true_id,
                device=device,
            )
            candidates = torch.cat(
                [torch.tensor([true_id], device=device), distractors]
            )

            prefix_ids = true_ids_list[:token_idx]
            prefix_cache = None
            if prefix_ids:
                prefix_tensor = torch.tensor(prefix_ids, device=device).unsqueeze(0)
                with torch.no_grad():
                    prefix_output = model(prefix_tensor, use_cache=True)
                prefix_cache = prefix_output.past_key_values

            losses = []
            for start in range(0, len(candidates), batch_size):
                end = min(start + batch_size, len(candidates))
                batch_candidates = candidates[start:end].unsqueeze(1)
                current_batch = end - start

                with torch.no_grad():
                    if prefix_cache is not None:
                        expanded_cache = expand_past_key_values(prefix_cache, current_batch)
                        hidden = model(
                            batch_candidates,
                            past_key_values=expanded_cache,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]
                    else:
                        hidden = model(
                            batch_candidates,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]

                batch_losses = ((hidden - target_hidden.unsqueeze(0)) ** 2).mean(dim=1)
                losses.append(batch_losses.cpu())

            losses = torch.cat(losses).numpy()
            true_loss = float(losses[0])
            random_losses = losses[1:]
            rank = int((losses < true_loss).sum() + 1)

            results.append(
                {
                    "tok_idx": token_idx,
                    "true_id": true_id,
                    "true_token": tokenizer.decode([true_id]),
                    "true_loss": true_loss,
                    "rand_losses": random_losses,
                    "rank": rank,
                    "min_rand": float(random_losses.min()) if len(random_losses) else float("nan"),
                    "median_rand": float(np.median(random_losses)) if len(random_losses) else float("nan"),
                    "sample_count": int(len(random_losses)),
                }
            )
            bar.update()

    landscape_prompt = prompt
    landscape_results = results
    return landscape_prompt, landscape_results


@app.cell
def _(landscape_prompt, landscape_results, mo, np, plt):
    if landscape_results is None:
        mo.md(
            "Enter a prompt above and click **▶ Inspect Loss Landscape**."
        ).callout(kind="info")
        return

    if len(landscape_results) == 0:
        mo.md("Prompt tokenized to zero tokens. Try a different prompt.").callout(
            kind="danger"
        )
        return

    results = landscape_results
    token_count = len(results)
    distractor_count = results[0]["sample_count"]

    ncols = min(token_count, 6)
    fig1, axes = plt.subplots(
        1, ncols, figsize=(3.0 * ncols, 3.8), constrained_layout=True
    )
    if ncols == 1:
        axes = [axes]

    for result, axis in zip(results[:ncols], axes):
        all_losses = np.concatenate([[result["true_loss"]], result["rand_losses"]])
        sorted_idx = np.argsort(all_losses)
        top30 = sorted_idx[:30]
        colors = ["#2ecc71" if idx == 0 else "#3498db" for idx in top30]
        values = all_losses[top30]
        plotted_values = np.where(values <= 0, 1e-15, values)

        axis.bar(
            range(len(top30)),
            plotted_values,
            color=colors,
            alpha=0.85,
            width=0.8,
        )
        axis.set_yscale("log")
        axis.set_xlabel("Top sampled ranks")
        axis.set_ylabel("MSE loss (log)")
        token_display = repr(result["true_token"]).strip("'")
        axis.set_title(
            f"Position {result['tok_idx']} -> `{token_display}`\n"
            f"True loss: {result['true_loss']:.1e}   Rank: {result['rank']}",
            fontsize=9,
        )
        axis.set_xticks([])

    fig2, axis2 = plt.subplots(
        figsize=(max(4, token_count * 1.2), 3.5), constrained_layout=True
    )
    ratios = [
        result["median_rand"] / max(result["true_loss"], 1e-15)
        for result in results
    ]
    token_labels = [repr(result["true_token"]).strip("'") for result in results]
    bars = axis2.bar(token_labels, ratios, color="#2ecc71", alpha=0.85)
    axis2.set_yscale("log")
    axis2.set_xlabel("Token")
    axis2.set_ylabel("Sampled margin ratio")
    axis2.set_title(
        "Sampled margin ratio = median distractor loss / true-token loss",
        fontsize=9.5,
    )
    axis2.grid(True, alpha=0.2, axis="y")
    for bar, value in zip(bars, ratios):
        axis2.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.3,
            f"{value:.0e}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    rows = "\n".join(
        f"| `{result['true_token']}` | {result['true_loss']:.2e} | "
        f"{result['min_rand']:.2e} | {result['median_rand']:.2e} | "
        f"**{result['rank']}** |"
        for result in results
    )
    table = (
        "| Token | True loss | Min distractor | Median distractor | Rank |\n"
        "|---|---:|---:|---:|---:|\n"
        + rows
    )

    all_rank1 = all(result["rank"] == 1 for result in results)
    summary_kind = "success" if all_rank1 else "warn"
    summary_text = (
        f"Across **{distractor_count} sampled distractors per position**, the true token ranked "
        "**#1** everywhere."
        if all_rank1
        else "At least one position did not rank the true token first inside the sampled set. "
        "Increase the distractor count or try a shorter prompt."
    )

    mo.vstack(
        [
            mo.md(f"### Prompt: `{landscape_prompt}`\n\n{summary_text}").callout(
                kind=summary_kind
            ),
            mo.md(
                "Green bars are the true tokens; blue bars are sampled distractors. "
                "This section visualizes **sampled** one-step separability, not a full-vocabulary proof."
            ),
            fig1,
            mo.md(
                "Higher sampled margin ratios mean the true token is much easier to identify than a "
                "typical sampled impostor."
            ),
            fig2,
            mo.md("### Per-position breakdown"),
            mo.md(table),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## § 3  Exact Prompt Recovery on GPT-2

        The paper's theory allows different candidate policies. This notebook implements the most
        transparent version: **full-vocabulary exhaustive search** at a fixed layer, one token at a
        time, using cached prefixes to keep the CPU runtime reasonable.

        Notes:

        - This is the hidden-state leakage setting from the paper.
        - The live run below is exact but slow, so keep prompts short.
        - The notebook uses CPU by default; start with 2 to 4 tokens.
        """
    )
    return


@app.cell
def _(mo):
    recover_input = mo.ui.text(
        value="Hello world how",
        label="Prompt to invert",
        full_width=True,
    )
    recover_layer_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=12,
        label="Recovery layer",
        show_value=True,
        full_width=True,
    )
    recover_batch_slider = mo.ui.slider(
        steps=[128, 256, 512, 1024],
        value=512,
        label="Candidate batch size",
        show_value=True,
        full_width=True,
    )
    recover_btn = mo.ui.run_button(label="▶ Recover Prompt Exactly (Full GPT-2 Vocabulary)")

    mo.vstack(
        [
            mo.md("### Controls"),
            recover_input,
            mo.hstack([recover_layer_slider, recover_batch_slider], widths="equal"),
            recover_btn,
        ],
        gap=0.6,
    )
    return recover_batch_slider, recover_btn, recover_input, recover_layer_slider


@app.cell
def _(mo, recover_input, tokenizer):
    ids = tokenizer.encode(recover_input.value or "")
    preview = "  ".join(f"`{tokenizer.decode([token])}`" for token in ids) if ids else "_empty_"
    mo.md(f"**Tokens ({len(ids)}):** {preview}")
    return


@app.cell
def _(
    device,
    expand_past_key_values,
    model,
    mo,
    recover_batch_slider,
    recover_btn,
    recover_input,
    recover_layer_slider,
    tokenizer,
    torch,
):
    if not recover_btn.value:
        recovery_prompt = None
        recovery_results = None
        recovery_config = None
        return recovery_prompt, recovery_results, recovery_config

    prompt = (recover_input.value or "").strip()
    layer_idx = int(recover_layer_slider.value)
    batch_size = int(recover_batch_slider.value)
    config = {
        "layer": layer_idx,
        "batch_size": batch_size,
        "status": "ok",
    }

    true_ids = tokenizer.encode(prompt)
    token_count = len(true_ids)
    vocab_size = model.wte.weight.shape[0]

    if token_count == 0:
        config["status"] = "empty"
        recovery_prompt = prompt
        recovery_results = []
        recovery_config = config
        return recovery_prompt, recovery_results, recovery_config

    if token_count > 6:
        config["status"] = "too_long"
        recovery_prompt = prompt
        recovery_results = []
        recovery_config = config
        return recovery_prompt, recovery_results, recovery_config

    targets = []
    with torch.no_grad():
        for position in range(token_count):
            prefix_tensor = torch.tensor(
                true_ids[: position + 1], device=device
            ).unsqueeze(0)
            target_output = model(prefix_tensor, output_hidden_states=True)
            targets.append(target_output.hidden_states[layer_idx][0, -1, :].detach())

    recovered = []
    results = []

    with mo.status.progress_bar(total=token_count, title="Recovering tokens...") as bar:
        for position in range(token_count):
            target_hidden = targets[position]

            prefix_cache = None
            if recovered:
                prefix_tensor = torch.tensor(recovered, device=device).unsqueeze(0)
                with torch.no_grad():
                    prefix_output = model(prefix_tensor, use_cache=True)
                prefix_cache = prefix_output.past_key_values

            best_token = 0
            best_loss = float("inf")

            for start in range(0, vocab_size, batch_size):
                end = min(start + batch_size, vocab_size)
                current_batch = end - start
                candidates = torch.arange(start, end, device=device).unsqueeze(1)

                with torch.no_grad():
                    if prefix_cache is not None:
                        expanded_cache = expand_past_key_values(prefix_cache, current_batch)
                        hidden = model(
                            candidates,
                            past_key_values=expanded_cache,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]
                    else:
                        hidden = model(
                            candidates,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]

                losses = ((hidden - target_hidden) ** 2).mean(dim=1)
                best_idx = int(losses.argmin())
                if float(losses[best_idx]) < best_loss:
                    best_loss = float(losses[best_idx])
                    best_token = start + best_idx

            recovered.append(best_token)
            results.append(
                {
                    "pos": position,
                    "true_id": true_ids[position],
                    "recovered_id": best_token,
                    "true_word": tokenizer.decode([true_ids[position]]),
                    "recovered_word": tokenizer.decode([best_token]),
                    "min_loss": best_loss,
                    "correct": best_token == true_ids[position],
                }
            )
            bar.update()

    recovery_prompt = prompt
    recovery_results = results
    recovery_config = config
    return recovery_prompt, recovery_results, recovery_config


@app.cell
def _(mo, recovery_config, recovery_prompt, recovery_results):
    if recovery_results is None:
        # Pre-computed on GPT-2 (layer 12, batch 512, no ln_f) — shown so the
        # result is visible immediately without waiting for the CPU run.
        prebaked = [
            {"pos": 0, "true_word": "Hello", "recovered_word": "Hello", "min_loss": 2.50e-08, "correct": True},
            {"pos": 1, "true_word": " world", "recovered_word": " world", "min_loss": 7.74e-11, "correct": True},
            {"pos": 2, "true_word": " how", "recovered_word": " how", "min_loss": 1.12e-10, "correct": True},
        ]
        pre_rows = "\n".join(
            f"| {r['pos']} | `{r['true_word']}` | `{r['recovered_word']}` "
            f"| {r['min_loss']:.2e} | {'✓' if r['correct'] else '✗'} |"
            for r in prebaked
        )
        pre_table = (
            "| Pos | True token | Recovered | Minimum loss | Match |\n"
            "|---|---|---|---:|:---:|\n" + pre_rows
        )
        mo.vstack(
            [
                mo.md(
                    "### Pre-computed: Recovery of `Hello world how`\n\n"
                    'Recovered **`"Hello world how"`** exactly from leaked GPT-2 '
                    "hidden states at layer 12."
                ).callout(kind="success"),
                mo.md(
                    "Run ahead of time to save you the wait. Enter your own prompt "
                    "above and click **▶ Recover Prompt Exactly** to try it live — "
                    "the search covers all 50,257 GPT-2 tokens at each position."
                ).callout(kind="neutral"),
                mo.md("### Per-position breakdown"),
                mo.md(pre_table),
            ]
        )
        return

    if recovery_config["status"] == "empty":
        mo.md("Prompt tokenized to zero tokens. Try a different input.").callout(
            kind="danger"
        )
        return

    if recovery_config["status"] == "too_long":
        mo.md(
            "This exact CPU demo is capped at 6 tokens. Shorten the prompt and run it again."
        ).callout(kind="warn")
        return

    results = recovery_results
    all_correct = all(result["correct"] for result in results)
    recovered_str = "".join(result["recovered_word"] for result in results)
    rows = "\n".join(
        f"| {result['pos']} | `{result['true_word']}` | `{result['recovered_word']}` "
        f"| {result['min_loss']:.2e} | {'✓' if result['correct'] else '✗'} |"
        for result in results
    )
    table = (
        "| Pos | True token | Recovered | Minimum loss | Match |\n"
        "|---|---|---|---:|:---:|\n"
        + rows
    )

    kind = "success" if all_correct else "danger"
    message = (
        f'Recovered **`"{recovered_str}"`** exactly from leaked hidden states.'
        if all_correct
        else f'Partial recovery for **`"{recovery_prompt}"`**. Inspect the table below.'
    )

    mo.vstack(
        [
            mo.md(
                f"### Recovery of `{recovery_prompt}`\n\n"
                f"Layer: **{recovery_config['layer']}**, batch size: **{recovery_config['batch_size']}**"
            ).callout(kind=kind),
            mo.md(message),
            mo.md(
                "This section performs a full search over all 50,257 GPT-2 tokens at each step. "
                "The smallest-loss token is selected as the reconstruction."
            ).callout(kind="neutral"),
            mo.md("### Per-position breakdown"),
            mo.md(table),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## § 4  What the Notebook Proves, and What It Does Not

        - The paper's proofs are about **almost-sure injectivity** of decoder-only transformers under
          continuous initialization and finite gradient-based training.
        - The practical attack assumes access to the **hidden-state sequence** at a fixed layer.
        - The brute-force GPT-2 section in this notebook is an exact verifier for that setting.
        - The PCA and sampled loss-landscape sections are illustrations, not proofs.
        - The paper's Theorem **3.2** is a **bounded-noise** statement: if the perturbation at each
          position stays below half the local separation margin, SIPIT still recovers the true tokens.

        Full GPT-2 noise sweeps are slow on CPU, so the extension below turns that half-margin idea
        into a small interactive demo.
        """
    ).callout(kind="info")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## § 5  Extension: Noise and Quantization Robustness on GPT-2

        Theorem 3.2 says recovery survives a perturbation if, at each position,
        the perturbation norm stays below half the local separation margin.

        This section tests that directly on real GPT-2 hidden states: take the same
        iterative targets used in § 3, corrupt them with additive isotropic noise,
        uniform bit-width quantization, or both — then re-run the full-vocabulary
        search on the corrupted targets.

        Start with noise = 0 and quant = off to confirm the clean baseline, then
        increase noise or lower bit-width to watch recovery break.
        Keep prompts short (≤ 4 tokens) for reasonable CPU runtime.
        """
    )
    return


@app.cell
def _(mo):
    robust_input = mo.ui.text(
        value="Hello world",
        label="Prompt to perturb",
        full_width=True,
    )
    robust_layer_slider = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=12,
        label="Target layer",
        show_value=True,
        full_width=True,
    )
    robust_batch_slider = mo.ui.slider(
        steps=[128, 256, 512, 1024],
        value=512,
        label="Candidate batch size",
        show_value=True,
        full_width=True,
    )
    robust_noise_slider = mo.ui.slider(
        start=0.0,
        stop=5.0,
        step=0.25,
        value=0.0,
        label="Noise radius (‖δ‖ per position)",
        show_value=True,
        full_width=True,
    )
    robust_quant_slider = mo.ui.slider(
        steps=[0, 4, 8],
        value=0,
        label="Quantization bits (0 = off)",
        show_value=True,
        full_width=True,
    )
    robust_seed_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=1,
        value=0,
        label="Noise seed",
        show_value=True,
        full_width=True,
    )
    robust_btn = mo.ui.run_button(label="▶ Run Perturbed Recovery")

    mo.vstack(
        [
            mo.md("### Controls"),
            robust_input,
            mo.hstack([robust_layer_slider, robust_batch_slider], widths="equal"),
            mo.hstack([robust_noise_slider, robust_quant_slider], widths="equal"),
            robust_seed_slider,
            robust_btn,
        ],
        gap=0.6,
    )
    return (
        robust_batch_slider,
        robust_btn,
        robust_input,
        robust_layer_slider,
        robust_noise_slider,
        robust_quant_slider,
        robust_seed_slider,
    )


@app.cell
def _(mo, robust_input, tokenizer):
    ids = tokenizer.encode(robust_input.value or "")
    preview = "  ".join(f"`{tokenizer.decode([token])}`" for token in ids) if ids else "_empty_"
    mo.md(f"**Tokens ({len(ids)}):** {preview}")
    return


@app.cell
def _(
    add_noise_to_tensor,
    device,
    expand_past_key_values,
    model,
    mo,
    quantize_tensor,
    robust_batch_slider,
    robust_btn,
    robust_input,
    robust_layer_slider,
    robust_noise_slider,
    robust_quant_slider,
    robust_seed_slider,
    tokenizer,
    torch,
):
    if not robust_btn.value:
        robust_prompt = None
        robust_results = None
        robust_config = None
        return robust_prompt, robust_results, robust_config

    prompt = (robust_input.value or "").strip()
    layer_idx = int(robust_layer_slider.value)
    batch_size = int(robust_batch_slider.value)
    noise_radius = float(robust_noise_slider.value)
    quant_bits = int(robust_quant_slider.value)
    seed = int(robust_seed_slider.value)

    config = {
        "layer": layer_idx,
        "batch_size": batch_size,
        "noise_radius": noise_radius,
        "quant_bits": quant_bits,
        "seed": seed,
        "status": "ok",
    }

    true_ids = tokenizer.encode(prompt)
    token_count = len(true_ids)
    vocab_size = model.wte.weight.shape[0]

    if token_count == 0:
        config["status"] = "empty"
        robust_prompt = prompt
        robust_results = []
        robust_config = config
        return robust_prompt, robust_results, robust_config

    if token_count > 6:
        config["status"] = "too_long"
        robust_prompt = prompt
        robust_results = []
        robust_config = config
        return robust_prompt, robust_results, robust_config

    # Compute clean iterative targets (same definition as § 3)
    clean_targets = []
    with torch.no_grad():
        for position in range(token_count):
            prefix_tensor = torch.tensor(
                true_ids[: position + 1], device=device
            ).unsqueeze(0)
            out = model(prefix_tensor, output_hidden_states=True)
            clean_targets.append(out.hidden_states[layer_idx][0, -1, :].detach())

    # Perturb: add noise then quantize
    perturbed_targets = []
    perturbation_norms = []
    for position, target in enumerate(clean_targets):
        noisy = add_noise_to_tensor(target, noise_radius, seed=seed * 1000 + position)
        quantized = quantize_tensor(noisy, quant_bits)
        perturbed_targets.append(quantized)
        perturbation_norms.append(float((quantized - target).norm()))

    # Full-vocabulary search on the perturbed targets
    recovered = []
    results = []

    with mo.status.progress_bar(
        total=token_count, title="Recovering from perturbed hidden states..."
    ) as bar:
        for position in range(token_count):
            target_hidden = perturbed_targets[position]

            prefix_cache = None
            if recovered:
                prefix_tensor = torch.tensor(recovered, device=device).unsqueeze(0)
                with torch.no_grad():
                    prefix_output = model(prefix_tensor, use_cache=True)
                prefix_cache = prefix_output.past_key_values

            best_token = 0
            best_loss = float("inf")

            for start in range(0, vocab_size, batch_size):
                end = min(start + batch_size, vocab_size)
                current_batch = end - start
                candidates = torch.arange(start, end, device=device).unsqueeze(1)

                with torch.no_grad():
                    if prefix_cache is not None:
                        expanded_cache = expand_past_key_values(
                            prefix_cache, current_batch
                        )
                        hidden = model(
                            candidates,
                            past_key_values=expanded_cache,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]
                    else:
                        hidden = model(
                            candidates,
                            output_hidden_states=True,
                        ).hidden_states[layer_idx][:, -1, :]

                losses = ((hidden - target_hidden) ** 2).mean(dim=1)
                best_idx = int(losses.argmin())
                if float(losses[best_idx]) < best_loss:
                    best_loss = float(losses[best_idx])
                    best_token = start + best_idx

            recovered.append(best_token)
            results.append(
                {
                    "pos": position,
                    "true_id": true_ids[position],
                    "recovered_id": best_token,
                    "true_word": tokenizer.decode([true_ids[position]]),
                    "recovered_word": tokenizer.decode([best_token]),
                    "perturbation_norm": perturbation_norms[position],
                    "min_loss": best_loss,
                    "correct": best_token == true_ids[position],
                }
            )
            bar.update()

    robust_prompt = prompt
    robust_results = results
    robust_config = config
    return robust_prompt, robust_results, robust_config


@app.cell
def _(mo, plt, robust_config, robust_prompt, robust_results):
    if robust_results is None:
        mo.md(
            "Enter a prompt above and click **▶ Run Perturbed Recovery**. "
            "Start with noise = 0 and quant = off to confirm the clean baseline, "
            "then increase noise or lower bit-width to watch recovery fail."
        ).callout(kind="info")
        return

    if robust_config["status"] == "empty":
        mo.md("Prompt tokenized to zero tokens.").callout(kind="danger")
        return

    if robust_config["status"] == "too_long":
        mo.md("Capped at 6 tokens. Shorten the prompt.").callout(kind="warn")
        return

    results = robust_results
    all_correct = all(r["correct"] for r in results)
    recovered_str = "".join(r["recovered_word"] for r in results)

    noise_str = f"noise radius {robust_config['noise_radius']:.2f}"
    quant_str = (
        f"{robust_config['quant_bits']}-bit quantization"
        if robust_config["quant_bits"] > 0
        else "no quantization"
    )
    kind = "success" if all_correct else "danger"
    message = (
        f"Recovery **succeeded** with {noise_str}, {quant_str}. "
        f"Recovered: **`\"{recovered_str}\"`**"
        if all_correct
        else f"Recovery **failed** with {noise_str}, {quant_str}. "
        f"Recovered `\"{recovered_str}\"` (expected `\"{robust_prompt}\"`)"
    )

    rows = "\n".join(
        f"| {r['pos']} | `{r['true_word']}` | `{r['recovered_word']}` "
        f"| {r['perturbation_norm']:.4f} | {r['min_loss']:.2e} | {'✓' if r['correct'] else '✗'} |"
        for r in results
    )
    table = (
        "| Pos | True token | Recovered | Perturbation ‖δ‖ | Min loss | Match |\n"
        "|---|---|---|---:|---:|:---:|\n" + rows
    )

    fig, ax = plt.subplots(
        figsize=(max(4.0, len(results) * 1.5), 3.5), constrained_layout=True
    )
    token_labels = [repr(r["true_word"]).strip("'") for r in results]
    norms = [r["perturbation_norm"] for r in results]
    bar_colors = ["#2ecc71" if r["correct"] else "#e74c3c" for r in results]
    ax.bar(token_labels, norms, color=bar_colors, alpha=0.85)
    ax.set_xlabel("Token")
    ax.set_ylabel("Perturbation ‖δ‖")
    ax.set_title(
        "Per-position perturbation norm\n(green = recovered correctly, red = wrong)",
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.2, axis="y")

    mo.vstack(
        [
            mo.md(
                f"### Perturbed Recovery of `{robust_prompt}`\n\n{message}"
            ).callout(kind=kind),
            mo.md(
                "Theorem 3.2 guarantees exact recovery when the perturbation at each position "
                "is below half the local separation margin. Increase noise or lower the "
                "bit-width to watch recovery break."
            ).callout(kind="neutral"),
            fig,
            mo.md("### Per-position breakdown"),
            mo.md(table),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Takeaways

        - The paper's threat model is **hidden-state leakage**, not black-box prompt recovery from output text.
        - GPT-2 already shows the one-step separation structure the paper relies on.
        - Full-vocabulary inversion is exact but slow on CPU, so the notebook keeps that section optional.
        - § 5 tests Theorem 3.2 directly on GPT-2: noise and quantization applied to the real hidden
          states either survive recovery or break it, depending on how large the perturbation is
          relative to the model's local separation margin.
        """
    )
    return


if __name__ == "__main__":
    app.run()
