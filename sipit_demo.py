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


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from transformers import GPT2Model, GPT2Tokenizer
    from transformers.cache_utils import DynamicCache

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
    })

    return DynamicCache, GPT2Model, GPT2Tokenizer, PCA, mo, nn, np, plt, torch


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.Html("""
            <div style="background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                        color: white; padding: 2.5rem 2rem; border-radius: 16px;">
              <div style="font-size: 0.78rem; letter-spacing: 0.12em; opacity: 0.6;
                          text-transform: uppercase; margin-bottom: 0.6rem;">
                ICLR 2026 &nbsp;·&nbsp; alphaXiv × marimo competition
              </div>
              <h1 style="margin: 0 0 0.6rem; font-size: 2rem; font-weight: 700; line-height: 1.25;">
                Language Models Are Injective<br>and Hence Invertible
              </h1>
              <div style="opacity: 0.75; font-size: 0.95rem;">
                Nikolaou et al. &nbsp;·&nbsp;
                <a href="https://arxiv.org/abs/2510.15511"
                   style="color: #93c5fd; text-decoration: none;">arXiv:2510.15511</a>
              </div>
            </div>
            """),
            mo.md(
                r"""
                The paper proves that decoder-only language models are **almost surely injective**:
                distinct input sequences map to distinct hidden states. The consequence for security
                is direct — anyone with access to a model's internal hidden representations can
                reconstruct the original prompt token by token.

                This notebook makes that claim concrete on a real model:

                - **§ 1** — GPT-2 hidden states visualised in 2D. Type any sentence to add it live.
                - **§ 2** — One-step loss landscapes show the true token as the unique near-zero minimiser.
                - **§ 3** — Full-vocabulary brute-force inversion: all 50,257 GPT-2 tokens searched, exact recovery verified.
                - **§ 5** — Extension (Theorem 3.2): recovery under noise and quantization on real GPT-2 hidden states.
                """
            ),
        ],
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    sentence_input = mo.ui.text(
        value="",
        placeholder="Type a sentence and press Enter...",
        label="Add your own sentence to the plot",
        full_width=True,
    )
    sentence_input
    return (sentence_input,)


@app.cell(hide_code=True)
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
    _colors = plt.cm.tab10(np.linspace(0, 1, len(base_sentences)))

    for idx, (point, sentence) in enumerate(zip(xy_base, base_sentences)):
        ax.scatter(*point, s=160, color=_colors[idx], zorder=5)
        ax.annotate(
            f'"{sentence}"',
            point,
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.5,
            color=_colors[idx],
        )

    if custom:
        # Project the user's sentence into the existing PCA space (no refit)
        with torch.no_grad():
            _ids = tokenizer.encode(custom, return_tensors="pt").to(device)
            _out = model(_ids, output_hidden_states=True)
            _h = _out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
        xy_custom = pca.transform(_h.reshape(1, -1))[0]
        ax.scatter(*xy_custom, s=280, color="black", marker="*", zorder=10,
                   label="★  Your sentence")
        ax.annotate(
            f'"{custom}"',
            xy_custom,
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8.5,
            color="black",
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.8)

    n_pairs = len(base_sentences) * (len(base_sentences) - 1) // 2
    ax.set_title(
        "GPT-2 last-layer hidden states — PCA projection (mean pooled)\n"
        f"Minimum pairwise L2 distance: {min_pairwise_dist:.1f}",
        fontsize=10.5,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variance explained)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} variance explained)")
    ax.grid(True, alpha=0.2)

    _note = (
        f" The black star is your sentence — **`\"{custom}\"`**." if custom else
        " Type a sentence above to add it as a black star."
    )
    mo.vstack(
        [
            fig,
            mo.md(
                f"Minimum pairwise L2 distance across {n_pairs} prompt pairs: "
                f"**{min_pairwise_dist:.2f}**. Each sentence lands at a distinct point — "
                f"consistent with injectivity, though not a proof of it.{_note}"
            ).callout(kind="neutral"),
        ]
    )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo, prompt_input, tokenizer):
    _ids = tokenizer.encode(prompt_input.value or "")
    _preview = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_preview}")
    return


@app.cell(hide_code=True)
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
    landscape_prompt = None
    landscape_results = None

    if run_btn.value:
        _prompt = (prompt_input.value or "").strip()
        _layer_idx = int(layer_slider.value)
        _sample_count = int(n_sample_slider.value)

        _true_ids_list = tokenizer.encode(_prompt)
        if not _true_ids_list:
            landscape_prompt = _prompt
            landscape_results = []
        else:
            _true_ids = torch.tensor(_true_ids_list).unsqueeze(0).to(device)
            _vocab_size = model.wte.weight.shape[0]
            _batch_size = 512

            with torch.no_grad():
                _target_output = model(_true_ids, output_hidden_states=True)
            _target_all = _target_output.hidden_states[_layer_idx][0].detach()

            _results = []
            with mo.status.progress_bar(total=len(_true_ids_list), title="Computing loss landscapes...") as _bar:
                for _token_idx in range(len(_true_ids_list)):
                    _target_hidden = _target_all[_token_idx]
                    _true_id = _true_ids_list[_token_idx]

                    _distractors = sample_without_true(
                        vocab_size=_vocab_size,
                        sample_size=_sample_count,
                        true_id=_true_id,
                        device=device,
                    )
                    _candidates = torch.cat(
                        [torch.tensor([_true_id], device=device), _distractors]
                    )

                    _prefix_ids = _true_ids_list[:_token_idx]
                    _prefix_cache = None
                    if _prefix_ids:
                        _prefix_tensor = torch.tensor(_prefix_ids, device=device).unsqueeze(0)
                        with torch.no_grad():
                            _prefix_output = model(_prefix_tensor, use_cache=True)
                        _prefix_cache = _prefix_output.past_key_values

                    _losses = []
                    for _start in range(0, len(_candidates), _batch_size):
                        _end = min(_start + _batch_size, len(_candidates))
                        _batch_candidates = _candidates[_start:_end].unsqueeze(1)
                        _current_batch = _end - _start

                        with torch.no_grad():
                            if _prefix_cache is not None:
                                _expanded_cache = expand_past_key_values(_prefix_cache, _current_batch)
                                _hidden = model(
                                    _batch_candidates,
                                    past_key_values=_expanded_cache,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]
                            else:
                                _hidden = model(
                                    _batch_candidates,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]

                        _batch_losses = ((_hidden - _target_hidden.unsqueeze(0)) ** 2).mean(dim=1)
                        _losses.append(_batch_losses.cpu())

                    _losses = torch.cat(_losses).numpy()
                    _true_loss = float(_losses[0])
                    _random_losses = _losses[1:]
                    _rank = int((_losses < _true_loss).sum() + 1)

                    _results.append(
                        {
                            "tok_idx": _token_idx,
                            "true_id": _true_id,
                            "true_token": tokenizer.decode([_true_id]),
                            "true_loss": _true_loss,
                            "rand_losses": _random_losses,
                            "rank": _rank,
                            "min_rand": float(_random_losses.min()) if len(_random_losses) else float("nan"),
                            "median_rand": float(np.median(_random_losses)) if len(_random_losses) else float("nan"),
                            "sample_count": int(len(_random_losses)),
                        }
                    )
                    _bar.update()

            landscape_prompt = _prompt
            landscape_results = _results

    return landscape_prompt, landscape_results


@app.cell(hide_code=True)
def _(landscape_prompt, landscape_results, mo, np, plt):
    if landscape_results is None:
        mo.md(
            "Enter a prompt above and click **▶ Inspect Loss Landscape**."
        ).callout(kind="info")
    elif len(landscape_results) == 0:
        mo.md("Prompt tokenized to zero tokens. Try a different prompt.").callout(
            kind="danger"
        )
    else:
        _results = landscape_results
        _token_count = len(_results)
        _distractor_count = _results[0]["sample_count"]

        _ncols = min(_token_count, 6)
        _fig1, _axes = plt.subplots(
            1, _ncols, figsize=(3.0 * _ncols, 3.8), constrained_layout=True
        )
        if _ncols == 1:
            _axes = [_axes]

        for _result, _axis in zip(_results[:_ncols], _axes):
            _all_losses = np.concatenate([[_result["true_loss"]], _result["rand_losses"]])
            _sorted_idx = np.argsort(_all_losses)
            _top30 = _sorted_idx[:30]
            _colors = ["#2ecc71" if _i == 0 else "#3498db" for _i in _top30]
            _values = _all_losses[_top30]
            _plotted_values = np.where(_values <= 0, 1e-15, _values)

            _axis.bar(
                range(len(_top30)),
                _plotted_values,
                color=_colors,
                alpha=0.85,
                width=0.8,
            )
            _axis.set_yscale("log")
            _axis.set_xlabel("Top sampled ranks")
            _axis.set_ylabel("MSE loss (log)")
            _token_display = repr(_result["true_token"]).strip("'")
            _axis.set_title(
                f"Position {_result['tok_idx']} → `{_token_display}`\n"
                f"True loss: {_result['true_loss']:.1e}   Rank: {_result['rank']}",
                fontsize=9,
            )
            _axis.set_xticks([])

        _fig2, _axis2 = plt.subplots(
            figsize=(max(4, _token_count * 1.2), 3.5), constrained_layout=True
        )
        _ratios = [
            _r["median_rand"] / max(_r["true_loss"], 1e-15)
            for _r in _results
        ]
        _token_labels = [repr(_r["true_token"]).strip("'") for _r in _results]
        _bars = _axis2.bar(_token_labels, _ratios, color="#2ecc71", alpha=0.85)
        _axis2.set_yscale("log")
        _axis2.set_xlabel("Token")
        _axis2.set_ylabel("Sampled margin ratio")
        _axis2.set_title(
            "Sampled margin ratio = median distractor loss / true-token loss",
            fontsize=9.5,
        )
        _axis2.grid(True, alpha=0.2, axis="y")
        for _bar, _val in zip(_bars, _ratios):
            _axis2.text(
                _bar.get_x() + _bar.get_width() / 2,
                _val * 1.3,
                f"{_val:.0e}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

        _rows = "\n".join(
            f"| `{_r['true_token']}` | {_r['true_loss']:.2e} | "
            f"{_r['min_rand']:.2e} | {_r['median_rand']:.2e} | "
            f"**{_r['rank']}** |"
            for _r in _results
        )
        _table = (
            "| Token | True loss | Min distractor | Median distractor | Rank |\n"
            "|---|---:|---:|---:|---:|\n"
            + _rows
        )

        _all_rank1 = all(_r["rank"] == 1 for _r in _results)
        _summary_kind = "success" if _all_rank1 else "warn"
        _summary_text = (
            f"Across **{_distractor_count} sampled distractors per position**, the true token ranked "
            "**#1** everywhere."
            if _all_rank1
            else "At least one position did not rank the true token first inside the sampled set. "
            "Increase the distractor count or try a shorter prompt."
        )

        mo.vstack(
            [
                mo.md(f"### Prompt: `{landscape_prompt}`\n\n{_summary_text}").callout(
                    kind=_summary_kind
                ),
                mo.md(
                    "Green bars are the true tokens; blue bars are sampled distractors. "
                    "This section visualizes **sampled** one-step separability, not a full-vocabulary proof."
                ),
                _fig1,
                mo.md(
                    "Higher sampled margin ratios mean the true token is much easier to identify than a "
                    "typical sampled impostor."
                ),
                _fig2,
                mo.md("### Per-position breakdown"),
                mo.md(_table),
            ]
        )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo, recover_input, tokenizer):
    _ids = tokenizer.encode(recover_input.value or "")
    _preview = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_preview}")
    return


@app.cell(hide_code=True)
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
    recovery_prompt = None
    recovery_results = None
    recovery_config = None

    if recover_btn.value:
        _prompt = (recover_input.value or "").strip()
        _layer_idx = int(recover_layer_slider.value)
        _batch_size = int(recover_batch_slider.value)
        _config = {"layer": _layer_idx, "batch_size": _batch_size, "status": "ok"}

        _true_ids = tokenizer.encode(_prompt)
        _token_count = len(_true_ids)
        _vocab_size = model.wte.weight.shape[0]

        if _token_count == 0:
            _config["status"] = "empty"
            recovery_prompt = _prompt
            recovery_results = []
            recovery_config = _config
        elif _token_count > 6:
            _config["status"] = "too_long"
            recovery_prompt = _prompt
            recovery_results = []
            recovery_config = _config
        else:
            _targets = []
            with torch.no_grad():
                for _position in range(_token_count):
                    _prefix_tensor = torch.tensor(
                        _true_ids[: _position + 1], device=device
                    ).unsqueeze(0)
                    _target_output = model(_prefix_tensor, output_hidden_states=True)
                    _targets.append(_target_output.hidden_states[_layer_idx][0, -1, :].detach())

            _recovered = []
            _results = []

            with mo.status.progress_bar(total=_token_count, title="Recovering tokens...") as _bar:
                for _position in range(_token_count):
                    _target_hidden = _targets[_position]

                    _prefix_cache = None
                    if _recovered:
                        _prefix_tensor = torch.tensor(_recovered, device=device).unsqueeze(0)
                        with torch.no_grad():
                            _prefix_output = model(_prefix_tensor, use_cache=True)
                        _prefix_cache = _prefix_output.past_key_values

                    _best_token = 0
                    _best_loss = float("inf")

                    for _start in range(0, _vocab_size, _batch_size):
                        _end = min(_start + _batch_size, _vocab_size)
                        _current_batch = _end - _start
                        _candidates = torch.arange(_start, _end, device=device).unsqueeze(1)

                        with torch.no_grad():
                            if _prefix_cache is not None:
                                _expanded_cache = expand_past_key_values(_prefix_cache, _current_batch)
                                _hidden = model(
                                    _candidates,
                                    past_key_values=_expanded_cache,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]
                            else:
                                _hidden = model(
                                    _candidates,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]

                        _losses = ((_hidden - _target_hidden) ** 2).mean(dim=1)
                        _best_idx = int(_losses.argmin())
                        if float(_losses[_best_idx]) < _best_loss:
                            _best_loss = float(_losses[_best_idx])
                            _best_token = _start + _best_idx

                    _recovered.append(_best_token)
                    _results.append(
                        {
                            "pos": _position,
                            "true_id": _true_ids[_position],
                            "recovered_id": _best_token,
                            "true_word": tokenizer.decode([_true_ids[_position]]),
                            "recovered_word": tokenizer.decode([_best_token]),
                            "min_loss": _best_loss,
                            "correct": _best_token == _true_ids[_position],
                        }
                    )
                    _bar.update()

            recovery_prompt = _prompt
            recovery_results = _results
            recovery_config = _config

    return recovery_prompt, recovery_results, recovery_config


@app.cell(hide_code=True)
def _(mo, recovery_config, recovery_prompt, recovery_results):
    if recovery_results is None:
        prebaked = [
            {"pos": 0, "true_word": "Hello", "recovered_word": "Hello", "min_loss": 2.50e-08, "correct": True},
            {"pos": 1, "true_word": " world", "recovered_word": " world", "min_loss": 7.74e-11, "correct": True},
            {"pos": 2, "true_word": " how", "recovered_word": " how", "min_loss": 1.12e-10, "correct": True},
        ]
        mo.vstack(
            [
                mo.md(
                    "### Pre-computed: Recovery of `Hello world how`\n\n"
                    'Recovered **`"Hello world how"`** exactly from leaked GPT-2 '
                    "hidden states at layer 12."
                ).callout(kind="success"),
                mo.hstack(
                    [
                        mo.stat(
                            value=r["true_word"].strip(),
                            label=f"Position {r['pos']}",
                            caption=f"min MSE {r['min_loss']:.2e} · exact match",
                            bordered=True,
                        )
                        for r in prebaked
                    ],
                    justify="start",
                ),
                mo.md(
                    "Run ahead of time to save you the wait. Enter your own prompt "
                    "above and click **▶ Recover Prompt Exactly** to try it live — "
                    "the search covers all 50,257 GPT-2 tokens at each position."
                ).callout(kind="neutral"),
            ]
        )
    elif recovery_config["status"] == "empty":
        mo.md("Prompt tokenized to zero tokens. Try a different input.").callout(kind="danger")
    elif recovery_config["status"] == "too_long":
        mo.md(
            "This exact CPU demo is capped at 6 tokens. Shorten the prompt and run it again."
        ).callout(kind="warn")
    else:
        _results = recovery_results
        _all_correct = all(_r["correct"] for _r in _results)
        _recovered_str = "".join(_r["recovered_word"] for _r in _results)
        _rows = "\n".join(
            f"| {_r['pos']} | `{_r['true_word']}` | `{_r['recovered_word']}` "
            f"| {_r['min_loss']:.2e} | {'✓' if _r['correct'] else '✗'} |"
            for _r in _results
        )
        _table = (
            "| Pos | True token | Recovered | Minimum loss | Match |\n"
            "|---|---|---|---:|:---:|\n"
            + _rows
        )
        _kind = "success" if _all_correct else "danger"
        _message = (
            f'Recovered **`"{_recovered_str}"`** exactly from leaked hidden states.'
            if _all_correct
            else f'Partial recovery for **`"{recovery_prompt}"`**. Check the breakdown below.'
        )
        mo.vstack(
            [
                mo.md(
                    f"### Recovery of `{recovery_prompt}`\n\n"
                    f"Layer: **{recovery_config['layer']}** · batch size: **{recovery_config['batch_size']}**\n\n"
                    f"{_message}"
                ).callout(kind=_kind),
                mo.hstack(
                    [
                        mo.stat(
                            value=_r["recovered_word"].strip() or _r["recovered_word"],
                            label=f"Position {_r['pos']}",
                            caption=f"min MSE {_r['min_loss']:.2e} · {'✓' if _r['correct'] else '✗ wrong'}",
                            bordered=True,
                        )
                        for _r in _results
                    ],
                    justify="start",
                    wrap=True,
                ),
                mo.md(
                    "Full search over all 50,257 GPT-2 tokens at each position. "
                    "The minimum-MSE token is selected as the reconstruction."
                ).callout(kind="neutral"),
                mo.md("### Per-position breakdown"),
                mo.md(_table),
            ]
        )
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo, robust_input, tokenizer):
    _ids = tokenizer.encode(robust_input.value or "")
    _preview = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_preview}")
    return


@app.cell(hide_code=True)
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
    robust_prompt = None
    robust_results = None
    robust_config = None

    if robust_btn.value:
        _prompt = (robust_input.value or "").strip()
        _layer_idx = int(robust_layer_slider.value)
        _batch_size = int(robust_batch_slider.value)
        _noise_radius = float(robust_noise_slider.value)
        _quant_bits = int(robust_quant_slider.value)
        _seed = int(robust_seed_slider.value)

        _config = {
            "layer": _layer_idx,
            "batch_size": _batch_size,
            "noise_radius": _noise_radius,
            "quant_bits": _quant_bits,
            "seed": _seed,
            "status": "ok",
        }

        _true_ids = tokenizer.encode(_prompt)
        _token_count = len(_true_ids)
        _vocab_size = model.wte.weight.shape[0]

        if _token_count == 0:
            _config["status"] = "empty"
            robust_prompt = _prompt
            robust_results = []
            robust_config = _config
        elif _token_count > 6:
            _config["status"] = "too_long"
            robust_prompt = _prompt
            robust_results = []
            robust_config = _config
        else:
            _clean_targets = []
            with torch.no_grad():
                for _position in range(_token_count):
                    _prefix_tensor = torch.tensor(
                        _true_ids[: _position + 1], device=device
                    ).unsqueeze(0)
                    _out = model(_prefix_tensor, output_hidden_states=True)
                    _clean_targets.append(_out.hidden_states[_layer_idx][0, -1, :].detach())

            _perturbed_targets = []
            _perturbation_norms = []
            for _position, _target in enumerate(_clean_targets):
                _noisy = add_noise_to_tensor(_target, _noise_radius, seed=_seed * 1000 + _position)
                _quantized = quantize_tensor(_noisy, _quant_bits)
                _perturbed_targets.append(_quantized)
                _perturbation_norms.append(float((_quantized - _target).norm()))

            _recovered = []
            _results = []

            with mo.status.progress_bar(
                total=_token_count, title="Recovering from perturbed hidden states..."
            ) as _bar:
                for _position in range(_token_count):
                    _target_hidden = _perturbed_targets[_position]

                    _prefix_cache = None
                    if _recovered:
                        _prefix_tensor = torch.tensor(_recovered, device=device).unsqueeze(0)
                        with torch.no_grad():
                            _prefix_output = model(_prefix_tensor, use_cache=True)
                        _prefix_cache = _prefix_output.past_key_values

                    _best_token = 0
                    _best_loss = float("inf")

                    for _start in range(0, _vocab_size, _batch_size):
                        _end = min(_start + _batch_size, _vocab_size)
                        _current_batch = _end - _start
                        _candidates = torch.arange(_start, _end, device=device).unsqueeze(1)

                        with torch.no_grad():
                            if _prefix_cache is not None:
                                _expanded_cache = expand_past_key_values(
                                    _prefix_cache, _current_batch
                                )
                                _hidden = model(
                                    _candidates,
                                    past_key_values=_expanded_cache,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]
                            else:
                                _hidden = model(
                                    _candidates,
                                    output_hidden_states=True,
                                ).hidden_states[_layer_idx][:, -1, :]

                        _losses = ((_hidden - _target_hidden) ** 2).mean(dim=1)
                        _best_idx = int(_losses.argmin())
                        if float(_losses[_best_idx]) < _best_loss:
                            _best_loss = float(_losses[_best_idx])
                            _best_token = _start + _best_idx

                    _recovered.append(_best_token)
                    _results.append(
                        {
                            "pos": _position,
                            "true_id": _true_ids[_position],
                            "recovered_id": _best_token,
                            "true_word": tokenizer.decode([_true_ids[_position]]),
                            "recovered_word": tokenizer.decode([_best_token]),
                            "perturbation_norm": _perturbation_norms[_position],
                            "min_loss": _best_loss,
                            "correct": _best_token == _true_ids[_position],
                        }
                    )
                    _bar.update()

            robust_prompt = _prompt
            robust_results = _results
            robust_config = _config

    return robust_prompt, robust_results, robust_config


@app.cell(hide_code=True)
def _(mo, plt, robust_config, robust_prompt, robust_results):
    if robust_results is None:
        _display = mo.md(
            "Enter a prompt above and click **▶ Run Perturbed Recovery**. "
            "Start with noise = 0 and quant = off to confirm the clean baseline, "
            "then increase noise or lower bit-width to watch recovery fail."
        ).callout(kind="info")
    elif robust_config["status"] == "empty":
        _display = mo.md("Prompt tokenized to zero tokens.").callout(kind="danger")
    elif robust_config["status"] == "too_long":
        _display = mo.md("Capped at 6 tokens. Shorten the prompt.").callout(kind="warn")
    else:
        _results = robust_results
        _all_correct = all(r["correct"] for r in _results)
        _recovered_str = "".join(r["recovered_word"] for r in _results)

        _noise_str = f"noise radius {robust_config['noise_radius']:.2f}"
        _quant_str = (
            f"{robust_config['quant_bits']}-bit quantization"
            if robust_config["quant_bits"] > 0
            else "no quantization"
        )
        _kind = "success" if _all_correct else "danger"
        _message = (
            f"Recovery **succeeded** with {_noise_str}, {_quant_str}. "
            f"Recovered: **`\"{_recovered_str}\"`**"
            if _all_correct
            else f"Recovery **failed** with {_noise_str}, {_quant_str}. "
            f"Recovered `\"{_recovered_str}\"` (expected `\"{robust_prompt}\"`)"
        )

        _rows = "\n".join(
            f"| {r['pos']} | `{r['true_word']}` | `{r['recovered_word']}` "
            f"| {r['perturbation_norm']:.4f} | {r['min_loss']:.2e} | {'✓' if r['correct'] else '✗'} |"
            for r in _results
        )
        _table = (
            "| Pos | True token | Recovered | Perturbation ‖δ‖ | Min loss | Match |\n"
            "|---|---|---|---:|---:|:---:|\n" + _rows
        )

        _fig, _ax = plt.subplots(
            figsize=(max(4.0, len(_results) * 1.5), 3.5), constrained_layout=True
        )
        _token_labels = [repr(r["true_word"]).strip("'") for r in _results]
        _norms = [r["perturbation_norm"] for r in _results]
        _bar_colors = ["#2ecc71" if r["correct"] else "#e74c3c" for r in _results]
        _ax.bar(_token_labels, _norms, color=_bar_colors, alpha=0.85)
        _ax.set_xlabel("Token")
        _ax.set_ylabel("Perturbation ‖δ‖")
        _ax.set_title(
            "Per-position perturbation norm\n(green = recovered correctly, red = wrong)",
            fontsize=9.5,
        )
        _ax.grid(True, alpha=0.2, axis="y")

        _display = mo.vstack(
            [
                mo.md(
                    f"### Perturbed Recovery of `{robust_prompt}`\n\n{_message}"
                ).callout(kind=_kind),
                mo.md(
                    "Theorem 3.2 guarantees exact recovery when the perturbation at each position "
                    "is below half the local separation margin. Increase noise or lower the "
                    "bit-width to watch recovery break."
                ).callout(kind="neutral"),
                _fig,
                mo.md("### Per-position breakdown"),
                mo.md(_table),
            ]
        )
    _display
    return


@app.cell(hide_code=True)
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
