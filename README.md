# Language Models Are Injective — and Hence Invertible

A marimo notebook walking through *Language Models are Injective and Hence Invertible* (Nikolaou et al., ICLR 2026; [arXiv:2510.15511](https://arxiv.org/abs/2510.15511)) for the alphaXiv × marimo notebook competition.

The notebook makes the paper's central claim tangible: a real GPT-2 model is run live, its hidden states are extracted, and the original prompt tokens are recovered exactly by brute-force search over the full 50,257-token vocabulary. No toy model, no approximation — the inversion actually works because the hidden-state map is injective.

## Run it

In molab (browser, no install):

```
https://molab.marimo.io/github/PeterLi-jpg/sipit-marimo/blob/main/sipit_demo.py/wasm?mode=read&show-code=false
```

Locally (no virtual env needed — `uvx` resolves dependencies inline):

```bash
uvx --with numpy --with matplotlib marimo run sipit_demo.py
```

The notebook ships with every GPT-2 hidden state, loss landscape, and
recovery result **pre-computed and embedded as data** (~125 KB JSON inside
the notebook), so it has **no torch / transformers dependency** and loads
instantly in WASM. Only `marimo`, `numpy`, and `matplotlib` are needed at
run time.

## Files

- `sipit_demo.py` — the notebook (single self-contained file with inline dependency metadata)

## What it covers

**§ 1 — Visual sanity check (interactive).** GPT-2 hidden states for eight sentences projected to 2D via PCA. Type any sentence into the text input and watch it appear on the plot instantly — the PCA axes are fixed, so the new point lands in the existing space.

**§ 2 — Loss landscape.** Samples distractor tokens and compares their hidden states to the true token's. The true token is the unique near-zero minimizer; all distractors are clearly separated. Sampled over a user-chosen number of candidates at a user-chosen layer.

**§ 3 — Exact prompt recovery (pre-computed result visible on load).** Full-vocabulary exhaustive search: for each token position, every GPT-2 token is evaluated as a candidate and the minimum-MSE token is selected. Recovery is exact — minimum loss values are on the order of 1e-10. The pre-baked result for `"Hello world how"` is shown immediately; click the run button to try your own prompt.

**§ 4 — Honest scope.** What the paper proves, what this notebook demonstrates, and what neither claims (e.g. recovering prompts from API output text).

**§ 5 — Extension: robustness to noise and quantization.** Applies additive isotropic noise and/or uniform bit-width quantization to the leaked hidden states, then re-runs the full-vocabulary recovery. Tests Theorem 3.2 directly on GPT-2: recovery survives small perturbations and breaks as the perturbation approaches the local separation margin. A full 5 × 3 robustness heatmap and phase-transition curve show all (noise, quantization) combinations at once, with the Theorem 3.2 half-margin bounds overlaid so the theory and the experiment land on the same axes.

**§ 6 — Extension: steganographic side channel.** Every correct recovery uses strictly less than the Theorem 3.2 noise budget. The leftover slack is a usable side channel: encode a secret payload as a perturbation δ with ‖δ‖ < ½ · margin, transmit the perturbed hidden state, and a receiver can recover *both* the original prompt (via SipIt) *and* the hidden message (by projecting the residual onto a shared orthonormal basis). The demo runs live in the browser — type any ASCII payload and watch encoding, decoding, and the per-position budget check update in real time. The channel capacity is 8 bits (one ASCII character) per token position.

## Key result

At layer 12, for the prompt `"Hello world how"`:

| Position | True token | Recovered | Min loss |
|---|---|---|---:|
| 0 | `Hello` | `Hello` | 2.50e-08 |
| 1 | ` world` | ` world` | 7.74e-11 |
| 2 | ` how` | ` how` | 1.12e-10 |

Exact match at every position.

## Credit

Paper: Nikolaou et al. (ICLR 2026).
