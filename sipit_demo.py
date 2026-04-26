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
"""Language Models Are Injective — and Hence Invertible.

A marimo notebook walking through Nikolaou et al. (ICLR 2026, arXiv:2510.15511)
for the alphaXiv × marimo competition.

Pre-computed sections (§ 1 base PCA, § 2 loss landscape, § 3 recovery table)
load instantly without the model. Live sections (custom sentences, live search)
require GPT-2 and activate once the model cell finishes loading.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", app_title="LMs Are Injective")


# ── Imports ───────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sipit_toolkit as T
    return T, mo, np, plt


@app.cell(hide_code=True)
def _palette(T):
    TRUE_COLOR  = T.TRUE_COLOR
    DIST_COLOR  = T.DIST_COLOR
    WRONG_COLOR = T.WRONG_COLOR
    PCA_COLORS  = T.PCA_COLORS
    return DIST_COLOR, PCA_COLORS, TRUE_COLOR, WRONG_COLOR


@app.cell(hide_code=True)
def _rcparams(T, plt):
    plt.rcParams.update(T.RCPARAMS)
    return


# ── Model loading (auto-runs; pre-computed cells don't depend on this) ─────────

@app.cell(hide_code=True)
def _model_load(mo):
    import torch
    import torch.nn as nn
    from transformers import GPT2Model, GPT2Tokenizer
    from transformers.cache_utils import DynamicCache

    with mo.status.spinner(
        title="Loading GPT-2 (~2–4 min on first visit in WASM, fast locally)…"
    ):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model     = GPT2Model.from_pretrained("gpt2")
        # Remove final layer norm: makes loss landscapes numerically cleaner.
        # This is a notebook presentation choice, not a theorem requirement.
        model.ln_f = nn.Identity()
        model.eval()
        device = torch.device("cpu")
        model  = model.to(device)

    return DynamicCache, device, model, nn, tokenizer, torch


@app.cell(hide_code=True)
def _model_utils(DynamicCache, torch):
    """Pure-utility functions shared across §2, §3, §5."""

    def expand_past_key_values(past_key_values, batch_size):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "layers"):
            expanded = DynamicCache()
            for li, layer in enumerate(past_key_values.layers):
                expanded.update(
                    layer.keys.expand(batch_size, -1, -1, -1).contiguous(),
                    layer.values.expand(batch_size, -1, -1, -1).contiguous(),
                    li,
                )
            return expanded
        return tuple(
            (k.expand(batch_size, -1, -1, -1).contiguous(),
             v.expand(batch_size, -1, -1, -1).contiguous())
            for k, v in past_key_values
        )

    def sample_without_true(vocab_size, sample_size, true_id, device):
        sample_size = max(0, min(sample_size, vocab_size - 1))
        if sample_size == 0:
            return torch.empty((0,), dtype=torch.long, device=device)
        draw = torch.randperm(vocab_size - 1, device=device)[:sample_size]
        return draw + (draw >= true_id).long()

    def add_noise(tensor, radius, seed):
        if radius <= 0.0:
            return tensor.clone()
        g = torch.Generator()
        g.manual_seed(seed)
        n = torch.randn(tensor.shape, generator=g)
        return tensor + radius * n / (n.norm() + 1e-12)

    def quantize(tensor, bits):
        if bits <= 0:
            return tensor.clone()
        mx = tensor.abs().max()
        if mx == 0:
            return tensor.clone()
        levels = (2 ** bits) - 1
        scaled = (tensor / mx + 1.0) / 2.0
        return ((scaled * levels).round() / levels * 2.0 - 1.0) * mx

    return add_noise, expand_past_key_values, quantize, sample_without_true


# ── Title & intro ──────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _title(mo):
    mo.Html("""
    <div style="background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
                color:white; padding:2.5rem 2rem 2rem; border-radius:16px;">
      <div style="font-size:0.75rem; letter-spacing:0.14em; opacity:0.55;
                  text-transform:uppercase; margin-bottom:0.5rem;">
        ICLR 2026 &nbsp;·&nbsp; alphaXiv × marimo competition
      </div>
      <h1 style="margin:0 0 0.5rem; font-size:2rem; font-weight:700; line-height:1.2;">
        Language Models Are Injective<br>and Hence Invertible
      </h1>
      <div style="opacity:0.7; font-size:0.92rem;">
        Nikolaou et al. &nbsp;·&nbsp;
        <a href="https://arxiv.org/abs/2510.15511"
           style="color:#93c5fd; text-decoration:none;">arXiv:2510.15511</a>
      </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _intro(mo):
    mo.vstack([
        mo.md(
            "The paper proves that decoder-only language models are **almost surely injective**: "
            "distinct input sequences map to distinct hidden states. Anyone with access to a "
            "model's internal representations can reconstruct the original prompt token by token."
        ),
        mo.Html("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.6rem; margin-top:0.25rem;">
          <div style="background:#eff6ff; border-radius:8px; padding:0.65rem 0.9rem;">
            <span style="font-weight:700; color:#3b82f6;">§ 1</span>
            <span style="font-size:0.88rem; color:#374151; margin-left:0.4rem;">
              Hidden states in 2D — add your own sentence live.
            </span>
          </div>
          <div style="background:#f5f3ff; border-radius:8px; padding:0.65rem 0.9rem;">
            <span style="font-weight:700; color:#8b5cf6;">§ 2</span>
            <span style="font-size:0.88rem; color:#374151; margin-left:0.4rem;">
              Loss landscapes — true token is the unique near-zero minimizer.
            </span>
          </div>
          <div style="background:#ecfeff; border-radius:8px; padding:0.65rem 0.9rem;">
            <span style="font-weight:700; color:#0891b2;">§ 3</span>
            <span style="font-size:0.88rem; color:#374151; margin-left:0.4rem;">
              Full-vocab inversion — all 50,257 tokens searched, exact match.
            </span>
          </div>
          <div style="background:#ecfdf5; border-radius:8px; padding:0.65rem 0.9rem;">
            <span style="font-weight:700; color:#10b981;">§ 5</span>
            <span style="font-size:0.88rem; color:#374151; margin-left:0.4rem;">
              Robustness — noise &amp; quantization vs Theorem 3.2.
            </span>
          </div>
        </div>
        """),
    ], gap=0.75)
    return


# ── § 1  PCA Sanity Check ──────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _s1_header(mo):
    mo.Html("""
    <div style="border-left:4px solid #3b82f6; padding:0.75rem 1.25rem;
                background:#eff6ff; border-radius:0 10px 10px 0; margin-top:1.5rem;">
      <span style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#3b82f6; font-weight:700;">§ 1</span>
      <h2 style="margin:0.15rem 0 0.35rem; color:#1e3a5f; font-size:1.25rem; font-weight:700;">
        Injectivity in Practice: A Visual Sanity Check
      </h2>
      <p style="margin:0; color:#374151; font-size:0.88rem; line-height:1.6;">
        Theorem 2.2 (almost-sure injectivity at initialization) and Theorem 2.3 (injectivity
        preserved under training) guarantee distinct sequences map to distinct hidden states.
        The 2-D PCA projection below makes this concrete on real GPT-2 geometry.
        The base sentences load from pre-computed data instantly.
        Type a sentence to add it live once GPT-2 is ready.
      </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _s1_sentence_ui(mo):
    sentence_input = mo.ui.text(
        value="",
        placeholder="Type a sentence and press Enter…",
        label="Add your own sentence to the plot",
        full_width=True,
    )
    sentence_input
    return (sentence_input,)


@app.cell(hide_code=True)
def _s1_custom_vec(T, device, model, np, sentence_input, tokenizer, torch):
    """Project a custom sentence into the pre-fitted PCA space (requires GPT-2)."""
    custom = (sentence_input.value or "").strip()
    xy_custom = None
    if custom:
        with torch.no_grad():
            _ids = tokenizer.encode(custom, return_tensors="pt").to(device)
            _out = model(_ids, output_hidden_states=True)
            _h   = _out.hidden_states[-1].squeeze(0).mean(0).cpu().numpy()
        xy_custom = T.pca_project(_h)
    return custom, xy_custom


@app.cell(hide_code=True)
def _s1_plot(PCA_COLORS, T, custom, mo, np, plt, xy_custom):
    def _draw():
        fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)

        for i, (pt, label) in enumerate(zip(T.XY_BASE, T.BASE_SENTENCES)):
            color = PCA_COLORS[i % len(PCA_COLORS)]
            ax.scatter(*pt, s=160, color=color, zorder=5)
            ax.annotate(
                f'"{label}"', pt,
                xytext=(7, 5), textcoords="offset points",
                fontsize=8.5, color=color,
            )

        if xy_custom is not None:
            ax.scatter(*xy_custom, s=280, color="#111", marker="*", zorder=10,
                       label="★  Your sentence")
            ax.annotate(
                f'"{custom}"', xy_custom,
                xytext=(7, 5), textcoords="offset points",
                fontsize=8.5, color="#111", fontweight="bold",
            )
            ax.legend(loc="lower right", fontsize=8.5, framealpha=0.8)

        ev = T.PCA_EXPLAINED_VARIANCE_RATIO
        ax.set_title(
            f"GPT-2 layer-12 hidden states — PCA projection (mean-pooled)\n"
            f"Minimum pairwise L2 distance across {len(T.BASE_SENTENCES)*(len(T.BASE_SENTENCES)-1)//2}"
            f" pairs: {T.MIN_PAIRWISE_DIST:.1f}",
            fontsize=10.5,
        )
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} variance)")
        ax.set_ylabel(f"PC2 ({ev[1]:.0%} variance)")
        ax.grid(True, alpha=0.2)
        return fig

    _note = (
        f" The black star is your sentence — **`\"{custom}\"`**." if custom else
        " Type a sentence above to add it as a black star (GPT-2 must be loaded)."
    )
    mo.vstack([
        mo.center(_draw()),
        mo.md(
            f"Each of the {len(T.BASE_SENTENCES)} sentences lands at a distinct point — "
            f"consistent with injectivity, though not a proof of it. {_note}"
        ).callout(kind="neutral"),
    ])
    return


# ── § 2  One-Step Loss Landscapes ─────────────────────────────────────────────

@app.cell(hide_code=True)
def _s2_header(mo):
    mo.Html("""
    <div style="border-left:4px solid #8b5cf6; padding:0.75rem 1.25rem;
                background:#f5f3ff; border-radius:0 10px 10px 0; margin-top:1.5rem;">
      <span style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#8b5cf6; font-weight:700;">§ 2</span>
      <h2 style="margin:0.15rem 0 0.35rem; color:#3b0764; font-size:1.25rem; font-weight:700;">
        Why Inversion Works: One-Step Loss Landscapes
      </h2>
      <p style="margin:0; color:#374151; font-size:0.88rem; line-height:1.6;">
        At each position the paper studies the map <em>v</em> ↦ <em>h<sub>t</sub></em>(π ⊕ v).
        If injective, only one candidate matches the observed hidden state. The pre-computed
        plot below shows this for <em>"The cat sat on the mat"</em>. Use the live controls
        to explore other prompts after GPT-2 loads.
      </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _s2_prebaked_plot(DIST_COLOR, TRUE_COLOR, T, mo, np, plt):
    """Show the pre-computed loss landscape — no model needed."""
    def _draw():
        results = T.LANDSCAPE_RESULTS
        ncols = len(results)
        fig, axes = plt.subplots(1, ncols, figsize=(3.0 * ncols, 3.8),
                                 constrained_layout=True)
        if ncols == 1:
            axes = [axes]
        for res, ax in zip(results, axes):
            colors = [TRUE_COLOR if is_t else DIST_COLOR
                      for is_t in res["top30_is_true"]]
            vals = np.array(res["top30_losses"])
            plotted = np.where(vals <= 0, 1e-15, vals)
            ax.bar(range(len(vals)), plotted, color=colors, alpha=0.85, width=0.8)
            ax.set_yscale("log")
            ax.set_title(
                f"Pos {res['tok_idx']} → `{res['true_token']}`\n"
                f"True: {res['true_loss']:.1e}   Rank: {res['rank']}",
                fontsize=9,
            )
            ax.set_xlabel("Top-30 by loss")
            ax.set_ylabel("MSE loss (log)")
            ax.set_xticks([])
        return fig

    _ratios = [r["median_rand"] / max(r["true_loss"], 1e-15)
               for r in T.LANDSCAPE_RESULTS]
    _labels = [r["true_token"] for r in T.LANDSCAPE_RESULTS]
    def _draw_ratios():
        fig, ax = plt.subplots(figsize=(max(4, len(_ratios) * 1.2), 3.2),
                               constrained_layout=True)
        bars = ax.bar(_labels, _ratios, color=TRUE_COLOR, alpha=0.85, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(top=max(_ratios) * 20)
        ax.set_xlabel("Token")
        ax.set_ylabel("Margin ratio (log)")
        ax.set_title(
            "Sampled margin ratio: median distractor ÷ true-token loss\n"
            "Higher = true token more isolated",
            fontsize=9.5,
        )
        ax.grid(True, alpha=0.25, axis="y", zorder=0)
        for bar, val in zip(bars, _ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 2.5,
                    f"{val:.0e}", ha="center", va="bottom", fontsize=7.5)
        return fig

    mo.vstack([
        mo.md(
            f"### Pre-computed: `{T.LANDSCAPE_PROMPT}` · layer {T.LANDSCAPE_LAYER}\n\n"
            "Green bar = true token · blue bars = sampled distractors. "
            "True token ranks **#1** at every position."
        ).callout(kind="success"),
        mo.center(_draw()),
        mo.md("Margin ratio: how much larger distractor losses are compared to the true token."),
        mo.center(_draw_ratios()),
    ])
    return


@app.cell(hide_code=True)
def _s2_controls(mo):
    prompt_input = mo.ui.text(
        value="The cat sat on the mat", label="Prompt", full_width=True,
    )
    layer_slider = mo.ui.slider(
        start=1, stop=12, step=1, value=12,
        label="Target layer", show_value=True, full_width=True,
    )
    n_sample_slider = mo.ui.slider(
        start=128, stop=2048, step=128, value=512,
        label="Distractor tokens per position", show_value=True, full_width=True,
    )
    run_btn = mo.ui.run_button(label="▶ Run Live Landscape")
    mo.vstack([
        mo.Html("<h4 style='margin:1rem 0 0.4rem; color:#6b7280;'>Live Controls</h4>"),
        prompt_input,
        mo.hstack([layer_slider, n_sample_slider], widths="equal"),
        run_btn,
    ], gap=0.5)
    return layer_slider, n_sample_slider, prompt_input, run_btn


@app.cell(hide_code=True)
def _s2_token_preview(mo, prompt_input, tokenizer):
    _ids = tokenizer.encode(prompt_input.value or "")
    _prev = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_prev}")
    return


@app.cell(hide_code=True)
def _s2_compute(
    device, expand_past_key_values, layer_slider,
    model, mo, n_sample_slider, np, prompt_input,
    run_btn, sample_without_true, tokenizer, torch,
):
    landscape_prompt = None
    landscape_results = None

    if run_btn.value:
        _prompt     = (prompt_input.value or "").strip()
        _layer_idx  = int(layer_slider.value)
        _n_sample   = int(n_sample_slider.value)
        _true_ids   = tokenizer.encode(_prompt)
        _vocab_size = model.wte.weight.shape[0]
        _batch_size = 512

        if not _true_ids:
            landscape_prompt  = _prompt
            landscape_results = []
        else:
            with torch.no_grad():
                _tgt_all = model(
                    torch.tensor(_true_ids).unsqueeze(0), output_hidden_states=True
                ).hidden_states[_layer_idx][0].detach()

            _results = []
            with mo.status.progress_bar(
                total=len(_true_ids), title="Computing loss landscapes…"
            ) as _bar:
                for _ti, _true_id in enumerate(_true_ids):
                    _tgt_h     = _tgt_all[_ti]
                    _distractors = sample_without_true(_vocab_size, _n_sample, _true_id, device)
                    _cands     = torch.cat([torch.tensor([_true_id], device=device), _distractors])
                    _pcache    = None
                    if _ti > 0:
                        with torch.no_grad():
                            _pcache = model(
                                torch.tensor(_true_ids[:_ti], device=device).unsqueeze(0),
                                use_cache=True,
                            ).past_key_values

                    _losses = []
                    for _s in range(0, len(_cands), _batch_size):
                        _e  = min(_s + _batch_size, len(_cands))
                        _bc = _cands[_s:_e].unsqueeze(1)
                        with torch.no_grad():
                            _kw = dict(output_hidden_states=True)
                            if _pcache is not None:
                                _kw["past_key_values"] = expand_past_key_values(_pcache, _e - _s)
                            _h = model(_bc, **_kw).hidden_states[_layer_idx][:, -1, :]
                        _losses.append(((_h - _tgt_h) ** 2).mean(1).cpu())

                    _all_l  = torch.cat(_losses).numpy()
                    _tl     = float(_all_l[0])
                    _rl     = _all_l[1:]
                    _sorted = np.argsort(_all_l)[:30]
                    _results.append({
                        "tok_idx": _ti,
                        "true_token": tokenizer.decode([int(_true_id)]),
                        "true_loss": _tl,
                        "rank": int((_all_l < _tl).sum() + 1),
                        "min_rand": float(_rl.min()) if len(_rl) else float("nan"),
                        "median_rand": float(np.median(_rl)) if len(_rl) else float("nan"),
                        "sample_count": len(_rl),
                        "top30_losses": _all_l[_sorted].tolist(),
                        "top30_is_true": [int(i == 0) for i in _sorted],
                    })
                    _bar.update()

            landscape_prompt  = _prompt
            landscape_results = _results

    return landscape_prompt, landscape_results


@app.cell(hide_code=True)
def _s2_live_plot(DIST_COLOR, TRUE_COLOR, landscape_prompt, landscape_results, mo, np, plt):
    if landscape_results is None:
        _display = mo.md(
            "Click **▶ Run Live Landscape** above to compute for a custom prompt."
        ).callout(kind="info")
    elif len(landscape_results) == 0:
        _display = mo.md("Prompt has zero tokens. Try a different input.").callout(kind="danger")
    else:
        _ncols = min(len(landscape_results), 6)
        _fig, _axes = plt.subplots(
            1, _ncols, figsize=(3.0 * _ncols, 3.8), constrained_layout=True
        )
        if _ncols == 1:
            _axes = [_axes]
        for _res, _ax in zip(landscape_results[:_ncols], _axes):
            _colors  = [TRUE_COLOR if _it else DIST_COLOR for _it in _res["top30_is_true"]]
            _vals    = np.array(_res["top30_losses"])
            _plotted = np.where(_vals <= 0, 1e-15, _vals)
            _ax.bar(range(len(_vals)), _plotted, color=_colors, alpha=0.85, width=0.8)
            _ax.set_yscale("log")
            _ax.set_title(
                f"Pos {_res['tok_idx']} → `{_res['true_token']}`\n"
                f"True: {_res['true_loss']:.1e}   Rank: {_res['rank']}",
                fontsize=9,
            )
            _ax.set_xlabel("Top-30 by loss")
            _ax.set_ylabel("MSE loss (log)")
            _ax.set_xticks([])

        _ratios = [r["median_rand"] / max(r["true_loss"], 1e-15) for r in landscape_results]
        _labels = [r["true_token"] for r in landscape_results]
        _fig2, _ax2 = plt.subplots(
            figsize=(max(4, len(_ratios) * 1.2), 3.2), constrained_layout=True
        )
        _bars = _ax2.bar(_labels, _ratios, color=TRUE_COLOR, alpha=0.85, zorder=3)
        _ax2.set_yscale("log")
        _ax2.set_ylim(top=max(_ratios) * 20)
        _ax2.set_xlabel("Token"); _ax2.set_ylabel("Margin ratio (log)")
        _ax2.set_title("Sampled margin ratio: median distractor ÷ true-token loss", fontsize=9.5)
        _ax2.grid(True, alpha=0.25, axis="y", zorder=0)
        for _bar, _val in zip(_bars, _ratios):
            _ax2.text(_bar.get_x() + _bar.get_width() / 2, _val * 2.5,
                      f"{_val:.0e}", ha="center", va="bottom", fontsize=7.5)

        _all_r1  = all(r["rank"] == 1 for r in landscape_results)
        _rows    = "\n".join(
            f"| `{r['true_token']}` | {r['true_loss']:.2e} | "
            f"{r['min_rand']:.2e} | {r['median_rand']:.2e} | **{r['rank']}** |"
            for r in landscape_results
        )
        _table = (
            "| Token | True loss | Min dist. | Median dist. | Rank |\n"
            "|---|---:|---:|---:|---:|\n" + _rows
        )
        _display = mo.vstack([
            mo.md(
                f"### Live: `{landscape_prompt}`\n\n"
                + ("True token ranked **#1** at every position." if _all_r1 else
                   "At least one position missed — try more distractors or a shorter prompt.")
            ).callout(kind="success" if _all_r1 else "warn"),
            mo.center(_fig),
            mo.center(_fig2),
            mo.md(_table),
        ])
    _display
    return


# ── § 3  Exact Prompt Recovery ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def _s3_header(mo):
    mo.Html("""
    <div style="border-left:4px solid #0891b2; padding:0.75rem 1.25rem;
                background:#ecfeff; border-radius:0 10px 10px 0; margin-top:1.5rem;">
      <span style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#0891b2; font-weight:700;">§ 3</span>
      <h2 style="margin:0.15rem 0 0.35rem; color:#0c4a6e; font-size:1.25rem; font-weight:700;">
        Exact Prompt Recovery on GPT-2
      </h2>
      <p style="margin:0; color:#374151; font-size:0.88rem; line-height:1.6;">
        Full-vocabulary exhaustive search at layer 12 — all 50,257 GPT-2 tokens evaluated
        at each position, minimum-MSE token selected. The pre-computed result below is exact.
        Click <strong>▶ Recover Exactly</strong> to run it live on a short prompt of your choice.
      </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _s3_prebaked(T, mo):
    _r = T.RECOVERY_PREBAKED
    mo.vstack([
        mo.md(
            "### Pre-computed: Recovery of `Hello world how`\n\n"
            "Recovered **`\"Hello world how\"`** exactly from GPT-2 layer-12 hidden states."
        ).callout(kind="success"),
        mo.hstack(
            [
                mo.stat(
                    value=row["true_word"].strip() or row["true_word"],
                    label=f"Position {row['pos']}",
                    caption=f"min MSE {row['min_loss']:.2e} · exact match",
                    bordered=True,
                )
                for row in _r
            ],
            justify="start",
        ),
        mo.md(
            "Pre-computed to save you the wait. "
            "Use the live controls below to verify your own short prompt."
        ).callout(kind="neutral"),
    ])
    return


@app.cell(hide_code=True)
def _s3_controls(mo):
    recover_input = mo.ui.text(
        value="Hello world how", label="Prompt to invert", full_width=True,
    )
    recover_layer = mo.ui.slider(
        start=1, stop=12, step=1, value=12,
        label="Recovery layer", show_value=True, full_width=True,
    )
    recover_batch = mo.ui.slider(
        steps=[128, 256, 512, 1024], value=1024,
        label="Candidate batch size", show_value=True, full_width=True,
    )
    recover_btn = mo.ui.run_button(label="▶ Recover Exactly (Full Vocabulary)")
    mo.vstack([
        mo.Html("<h4 style='margin:1rem 0 0.4rem; color:#6b7280;'>Live Controls</h4>"),
        recover_input,
        mo.hstack([recover_layer, recover_batch], widths="equal"),
        mo.md(
            "⏱ **Heads up:** full-vocab search over 50,257 tokens takes roughly "
            "**20–50 s per token** in WASM. Keep prompts to 1–3 tokens."
        ).callout(kind="warn"),
        recover_btn,
    ], gap=0.5)
    return recover_batch, recover_btn, recover_input, recover_layer


@app.cell(hide_code=True)
def _s3_token_preview(mo, recover_input, tokenizer):
    _ids = tokenizer.encode(recover_input.value or "")
    _prev = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_prev}")
    return


@app.cell(hide_code=True)
def _s3_compute(
    device, expand_past_key_values, model, mo,
    recover_batch, recover_btn, recover_input, recover_layer,
    tokenizer, torch,
):
    recovery_prompt  = None
    recovery_results = None
    recovery_config  = None

    if recover_btn.value:
        _prompt     = (recover_input.value or "").strip()
        _layer_idx  = int(recover_layer.value)
        _batch_size = int(recover_batch.value)
        _config     = {"layer": _layer_idx, "batch_size": _batch_size, "status": "ok"}
        _true_ids   = tokenizer.encode(_prompt)
        _vocab_size = model.wte.weight.shape[0]

        if not _true_ids:
            _config["status"] = "empty"
        elif len(_true_ids) > 6:
            _config["status"] = "too_long"
        else:
            _targets = []
            with torch.no_grad():
                for _pos in range(len(_true_ids)):
                    _out = model(
                        torch.tensor(_true_ids[:_pos + 1], device=device).unsqueeze(0),
                        output_hidden_states=True,
                    )
                    _targets.append(_out.hidden_states[_layer_idx][0, -1, :].detach())

            _recovered = []
            _results   = []
            with mo.status.progress_bar(
                total=len(_true_ids), title="Recovering tokens…"
            ) as _bar:
                for _pos in range(len(_true_ids)):
                    _tgt_h   = _targets[_pos]
                    _pcache  = None
                    if _recovered:
                        with torch.no_grad():
                            _pcache = model(
                                torch.tensor(_recovered, device=device).unsqueeze(0),
                                use_cache=True,
                            ).past_key_values

                    _best_tok  = 0
                    _best_loss = float("inf")
                    for _s in range(0, _vocab_size, _batch_size):
                        _e  = min(_s + _batch_size, _vocab_size)
                        _bc = torch.arange(_s, _e, device=device).unsqueeze(1)
                        with torch.no_grad():
                            _kw = dict(output_hidden_states=True)
                            if _pcache is not None:
                                _kw["past_key_values"] = expand_past_key_values(_pcache, _e - _s)
                            _h = model(_bc, **_kw).hidden_states[_layer_idx][:, -1, :]
                        _l   = ((_h - _tgt_h) ** 2).mean(1)
                        _bi  = int(_l.argmin())
                        if float(_l[_bi]) < _best_loss:
                            _best_loss = float(_l[_bi])
                            _best_tok  = _s + _bi

                    _recovered.append(_best_tok)
                    _results.append({
                        "pos": _pos,
                        "true_id": _true_ids[_pos],
                        "recovered_id": _best_tok,
                        "true_word": tokenizer.decode([_true_ids[_pos]]),
                        "recovered_word": tokenizer.decode([_best_tok]),
                        "min_loss": _best_loss,
                        "correct": _best_tok == _true_ids[_pos],
                    })
                    _bar.update()

            recovery_prompt  = _prompt
            recovery_results = _results
            recovery_config  = _config

        if _config["status"] != "ok":
            recovery_prompt  = _prompt
            recovery_results = []
            recovery_config  = _config

    return recovery_config, recovery_prompt, recovery_results


@app.cell(hide_code=True)
def _s3_live_plot(mo, recovery_config, recovery_prompt, recovery_results):
    if recovery_results is None:
        _display = mo.md(
            "Click **▶ Recover Exactly** above to run live on your own prompt."
        ).callout(kind="info")
    elif recovery_config["status"] == "empty":
        _display = mo.md("Zero tokens. Try a different prompt.").callout(kind="danger")
    elif recovery_config["status"] == "too_long":
        _display = mo.md(
            "Capped at 6 tokens for this demo. Shorten the prompt."
        ).callout(kind="warn")
    else:
        _all_ok  = all(r["correct"] for r in recovery_results)
        _rec_str = "".join(r["recovered_word"] for r in recovery_results)
        _rows    = "\n".join(
            f"| {r['pos']} | `{r['true_word']}` | `{r['recovered_word']}` "
            f"| {r['min_loss']:.2e} | {'✓' if r['correct'] else '✗'} |"
            for r in recovery_results
        )
        _table = (
            "| Pos | True token | Recovered | Min loss | Match |\n"
            "|---|---|---|---:|:---:|\n" + _rows
        )
        _display = mo.vstack([
            mo.md(
                f"### Recovery of `{recovery_prompt}`\n\n"
                f"Layer {recovery_config['layer']} · batch {recovery_config['batch_size']}\n\n"
                + (f'Recovered **`"{_rec_str}"`** exactly.' if _all_ok else
                   f'Partial match: **`"{_rec_str}"`**.')
            ).callout(kind="success" if _all_ok else "danger"),
            mo.hstack(
                [mo.stat(
                    value=r["recovered_word"].strip() or r["recovered_word"],
                    label=f"Position {r['pos']}",
                    caption=f"min MSE {r['min_loss']:.2e} · {'✓' if r['correct'] else '✗'}",
                    bordered=True,
                ) for r in recovery_results],
                justify="start", wrap=True,
            ),
            mo.md(_table),
        ])
    _display
    return


# ── § 4  Honest Scope ─────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _s4_scope(mo):
    mo.Html("""
    <div style="border-left:4px solid #f59e0b; padding:0.75rem 1.25rem;
                background:#fffbeb; border-radius:0 10px 10px 0; margin-top:1.5rem;">
      <span style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#d97706; font-weight:700;">§ 4 — Honest Scope</span>
      <h2 style="margin:0.15rem 0 0.35rem; color:#78350f; font-size:1.25rem; font-weight:700;">
        What the Notebook Proves, and What It Does Not
      </h2>
      <ul style="margin:0.25rem 0 0; color:#374151; font-size:0.88rem;
                 line-height:1.8; padding-left:1.2rem;">
        <li>The paper proves <strong>almost-sure injectivity</strong> of decoder-only
            transformers under continuous initialization and finite gradient training.</li>
        <li>The practical attack requires access to the <strong>hidden-state sequence</strong>
            at a fixed layer — not just output text from an API.</li>
        <li>§ 3 brute-force inversion is an exact verifier for that leakage setting.
            § 1 PCA and § 2 loss landscapes are illustrations, not proofs.</li>
        <li>Theorem <strong>3.2</strong> is a bounded-noise result: recovery survives if
            the perturbation at each position stays below half the local separation margin.
            § 5 tests this directly on GPT-2.</li>
      </ul>
    </div>
    """)
    return


# ── § 5  Noise & Quantization Robustness ──────────────────────────────────────

@app.cell(hide_code=True)
def _s5_header(mo):
    mo.Html("""
    <div style="border-left:4px solid #10b981; padding:0.75rem 1.25rem;
                background:#ecfdf5; border-radius:0 10px 10px 0; margin-top:1.5rem;">
      <span style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#10b981; font-weight:700;">§ 5 — Extension</span>
      <h2 style="margin:0.15rem 0 0.35rem; color:#064e3b; font-size:1.25rem; font-weight:700;">
        Noise and Quantization Robustness on GPT-2
      </h2>
      <p style="margin:0; color:#374151; font-size:0.88rem; line-height:1.6;">
        Theorem 3.2 guarantees recovery when the perturbation at each position stays below
        half the local separation margin. Corrupt the layer-12 targets with additive isotropic
        noise, uniform quantization, or both — then re-run full-vocabulary search.
        Start at noise = 0 / quant = off to confirm the clean baseline,
        then increase noise or lower bit-width to watch recovery break.
        Keep prompts ≤ 4 tokens.
      </p>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _s5_controls(mo):
    robust_input = mo.ui.text(
        value="Hello world", label="Prompt to perturb", full_width=True,
    )
    robust_layer = mo.ui.slider(
        start=1, stop=12, step=1, value=12,
        label="Target layer", show_value=True, full_width=True,
    )
    robust_batch = mo.ui.slider(
        steps=[128, 256, 512, 1024], value=512,
        label="Candidate batch size", show_value=True, full_width=True,
    )
    robust_noise = mo.ui.slider(
        start=0.0, stop=5.0, step=0.25, value=0.0,
        label="Noise radius ‖δ‖", show_value=True, full_width=True,
    )
    robust_quant = mo.ui.slider(
        steps=[0, 4, 8], value=0,
        label="Quantization bits (0 = off)", show_value=True, full_width=True,
    )
    robust_seed = mo.ui.slider(
        start=0, stop=20, step=1, value=0,
        label="Noise seed", show_value=True, full_width=True,
    )
    robust_btn = mo.ui.run_button(label="▶ Run Perturbed Recovery")
    mo.vstack([
        robust_input,
        mo.hstack([robust_layer, robust_batch], widths="equal"),
        mo.hstack([robust_noise, robust_quant], widths="equal"),
        robust_seed,
        robust_btn,
    ], gap=0.5)
    return (
        robust_batch, robust_btn, robust_input,
        robust_layer, robust_noise, robust_quant, robust_seed,
    )


@app.cell(hide_code=True)
def _s5_token_preview(mo, robust_input, tokenizer):
    _ids = tokenizer.encode(robust_input.value or "")
    _prev = "  ".join(f"`{tokenizer.decode([t])}`" for t in _ids) if _ids else "_empty_"
    mo.md(f"**Tokens ({len(_ids)}):** {_prev}")
    return


@app.cell(hide_code=True)
def _s5_compute(
    add_noise, device, expand_past_key_values, model, mo, quantize,
    robust_batch, robust_btn, robust_input, robust_layer,
    robust_noise, robust_quant, robust_seed,
    tokenizer, torch,
):
    robust_prompt  = None
    robust_results = None
    robust_config  = None

    if robust_btn.value:
        _prompt      = (robust_input.value or "").strip()
        _layer_idx   = int(robust_layer.value)
        _batch_size  = int(robust_batch.value)
        _noise_r     = float(robust_noise.value)
        _quant_bits  = int(robust_quant.value)
        _seed        = int(robust_seed.value)
        _config      = {
            "layer": _layer_idx, "batch_size": _batch_size,
            "noise_radius": _noise_r, "quant_bits": _quant_bits,
            "seed": _seed, "status": "ok",
        }
        _true_ids    = tokenizer.encode(_prompt)
        _vocab_size  = model.wte.weight.shape[0]

        if not _true_ids:
            _config["status"] = "empty"
        elif len(_true_ids) > 6:
            _config["status"] = "too_long"
        else:
            _clean = []
            with torch.no_grad():
                for _pos in range(len(_true_ids)):
                    _out = model(
                        torch.tensor(_true_ids[:_pos + 1], device=device).unsqueeze(0),
                        output_hidden_states=True,
                    )
                    _clean.append(_out.hidden_states[_layer_idx][0, -1, :].detach())

            _perturbed = []
            _pert_norms = []
            for _pos, _tgt in enumerate(_clean):
                _noisy = add_noise(_tgt, _noise_r, _seed * 1000 + _pos)
                _quant = quantize(_noisy, _quant_bits)
                _perturbed.append(_quant)
                _pert_norms.append(float((_quant - _tgt).norm()))

            _recovered = []
            _results   = []
            with mo.status.progress_bar(
                total=len(_true_ids), title="Recovering from perturbed hidden states…"
            ) as _bar:
                for _pos in range(len(_true_ids)):
                    _tgt_h  = _perturbed[_pos]
                    _pcache = None
                    if _recovered:
                        with torch.no_grad():
                            _pcache = model(
                                torch.tensor(_recovered, device=device).unsqueeze(0),
                                use_cache=True,
                            ).past_key_values

                    _best_tok  = 0
                    _best_loss = float("inf")
                    for _s in range(0, _vocab_size, _batch_size):
                        _e  = min(_s + _batch_size, _vocab_size)
                        _bc = torch.arange(_s, _e, device=device).unsqueeze(1)
                        with torch.no_grad():
                            _kw = dict(output_hidden_states=True)
                            if _pcache is not None:
                                _kw["past_key_values"] = expand_past_key_values(_pcache, _e - _s)
                            _h = model(_bc, **_kw).hidden_states[_layer_idx][:, -1, :]
                        _l  = ((_h - _tgt_h) ** 2).mean(1)
                        _bi = int(_l.argmin())
                        if float(_l[_bi]) < _best_loss:
                            _best_loss = float(_l[_bi])
                            _best_tok  = _s + _bi

                    _recovered.append(_best_tok)
                    _results.append({
                        "pos": _pos,
                        "true_word": tokenizer.decode([_true_ids[_pos]]),
                        "recovered_word": tokenizer.decode([_best_tok]),
                        "perturbation_norm": _pert_norms[_pos],
                        "min_loss": _best_loss,
                        "correct": _best_tok == _true_ids[_pos],
                    })
                    _bar.update()

            robust_prompt  = _prompt
            robust_results = _results
            robust_config  = _config

        if _config["status"] != "ok":
            robust_prompt  = _prompt
            robust_results = []
            robust_config  = _config

    return robust_config, robust_prompt, robust_results


@app.cell(hide_code=True)
def _s5_live_plot(TRUE_COLOR, WRONG_COLOR, mo, plt, robust_config, robust_prompt, robust_results):
    if robust_results is None:
        _display = mo.md(
            "Click **▶ Run Perturbed Recovery** above. "
            "Start at noise = 0 / quant = off for the clean baseline."
        ).callout(kind="info")
    elif robust_config["status"] == "empty":
        _display = mo.md("Zero tokens.").callout(kind="danger")
    elif robust_config["status"] == "too_long":
        _display = mo.md("Capped at 6 tokens. Shorten the prompt.").callout(kind="warn")
    else:
        _all_ok  = all(r["correct"] for r in robust_results)
        _rec_str = "".join(r["recovered_word"] for r in robust_results)
        _nr      = robust_config["noise_radius"]
        _qb      = robust_config["quant_bits"]
        _noise_s = f"noise ‖δ‖ = {_nr:.2f}"
        _quant_s = f"{_qb}-bit quant" if _qb else "no quant"

        _fig, _ax = plt.subplots(
            figsize=(max(4.0, len(robust_results) * 1.5), 3.5),
            constrained_layout=True,
        )
        _labels = [repr(r["true_word"]).strip("'") for r in robust_results]
        _norms  = [r["perturbation_norm"] for r in robust_results]
        _bcolors = [TRUE_COLOR if r["correct"] else WRONG_COLOR for r in robust_results]
        _ax.bar(_labels, _norms, color=_bcolors, alpha=0.85)
        _ax.set_xlabel("Token")
        _ax.set_ylabel("Perturbation ‖δ‖")
        _ax.set_title(
            "Per-position perturbation norm\n(green = correct, red = wrong)",
            fontsize=9.5,
        )
        _ax.grid(True, alpha=0.2, axis="y")

        _rows = "\n".join(
            f"| {r['pos']} | `{r['true_word']}` | `{r['recovered_word']}` "
            f"| {r['perturbation_norm']:.4f} | {r['min_loss']:.2e} | "
            f"{'✓' if r['correct'] else '✗'} |"
            for r in robust_results
        )
        _table = (
            "| Pos | True | Recovered | ‖δ‖ | Min loss | Match |\n"
            "|---|---|---|---:|---:|:---:|\n" + _rows
        )
        _display = mo.vstack([
            mo.md(
                f"### Perturbed recovery of `{robust_prompt}`\n\n"
                f"{_noise_s}, {_quant_s}\n\n"
                + (f"Recovery **succeeded** — `\"{_rec_str}\"`" if _all_ok else
                   f"Recovery **failed** — got `\"{_rec_str}\"`")
            ).callout(kind="success" if _all_ok else "danger"),
            mo.md(
                "Theorem 3.2: recovery survives when perturbation < half the local "
                "separation margin. Increase noise or lower bit-width to cross the threshold."
            ).callout(kind="neutral"),
            mo.center(_fig),
            mo.md(_table),
        ])
    _display
    return


# ── Takeaways ──────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _takeaways(mo):
    mo.Html("""
    <div style="background:linear-gradient(135deg,#0f0c29cc,#302b63cc,#24243ecc);
                color:white; padding:2rem; border-radius:14px; margin-top:1.5rem;">
      <div style="font-size:0.7rem; letter-spacing:0.12em; opacity:0.55;
                  text-transform:uppercase; margin-bottom:0.5rem;">Key Takeaways</div>
      <h2 style="margin:0 0 1rem; font-size:1.2rem; font-weight:700;">What We Showed</h2>
      <ul style="margin:0; padding-left:1.2rem; line-height:2; font-size:0.9rem; opacity:0.9;">
        <li>The threat model is <strong>hidden-state leakage</strong>, not black-box
            text recovery from an API.</li>
        <li>GPT-2 already exhibits the one-step separation structure the injectivity
            proof relies on.</li>
        <li>Full-vocabulary brute-force inversion is <strong>exact</strong> on real GPT-2
            hidden states — minimum MSE ~ 10<sup>−10</sup>.</li>
        <li>§ 5 stress-tests Theorem 3.2: small perturbations survive, large ones break
            recovery — exactly as the half-margin bound predicts.</li>
      </ul>
      <div style="margin-top:1.25rem; padding-top:1rem;
                  border-top:1px solid rgba(255,255,255,0.15);
                  font-size:0.78rem; opacity:0.5;">
        Notebook for the alphaXiv × marimo competition &nbsp;·&nbsp;
        Nikolaou et al., ICLR 2026 &nbsp;·&nbsp;
        <a href="https://arxiv.org/abs/2510.15511" style="color:#93c5fd;">arXiv:2510.15511</a>
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
