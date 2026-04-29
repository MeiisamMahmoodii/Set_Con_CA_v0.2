# Set-ConCA: Executive Summary (Verified Rerun)

## What Is Strong
* **Cross-family transfer:** Gemma-3 4B -> LLaMA-3 8B reaches **69.5% +/- 0.6pp** vs **25% chance**.
* **Causal steering:** Set-ConCA gains **+9.8pp** at `alpha=10`; weak-to-strong gains **+10.7pp**; random control degrades sharply.
* **Linear bridge sufficiency:** linear bridge reaches **69.3%**, while nonlinear MLP falls to **64.2%**.
* **Sparse reconstruction trade-off:** Set-ConCA still beats SAE-TopK on MSE (**0.1735** vs **0.1868**) at roughly matched sparsity.

## What Is Mixed or Weaker
* **Pointwise raw transfer:** SAE-TopK now beats Set-ConCA on raw overlap (**78.4%** vs **69.5%**).
* **Consistency loss:** in TopK mode, it changes transfer by only **+0.1pp**.
* **Corruption test:** transfer stays near **69%** even under full corruption; this does **not** support a collapse-to-chance claim.
* **Single-model interpretability:** Set-ConCA is competitive, but not clearly better than SAE-L1 or PCA on the proxy metrics.

## New Extended Diagnostics
* **SOTA-like extensions:** Procrustes **0.7302**, Ridge **0.7242**, CCA **0.7300**, NMF **0.8348**, ICA **0.1307**.
* **Layerwise proxy search:** best pseudo-layer pair is **early -> mid = 0.7413**.
* **Relative-depth mapping:** 60% depth lands at **mid -> mid = 0.7405**.
* **Steering by layer bucket:** late pseudo-layer bucket gives the strongest gain (**+0.1861** at `alpha=5`).
* **Multilingual benchmarks finalized:** final-pass matrices completed for **WMT14 fr-en** and **OPUS100 multi-en**, each with **7 models / 26 directed pairs**.
* **Final-pass multilingual means:** Set-ConCA averages **0.3802** (WMT14) and **0.3688** (OPUS100), while ConCA(S=1) is **0.3720 / 0.3725**.
* **Coverage now includes:** Qwen-3B/7B, Mistral-7B, Gemma-2-2B, Llama-3.2-1B/3B, and Phi-3.5-mini.

## Current Positioning
* **Best framing:** Set-ConCA is a credible set-based sparse concept method with strong cross-family transfer and steering evidence, plus a now-working multilingual benchmark path.
* **Unsafe framing now:** claiming consistency is essential, corruption proves semantic dependence, PCA-32 improves transfer, or Set-ConCA beats all strong baselines on raw WMT14 EN/FR overlap.

---
*Verified on RTX 3090 GPU | Full end-to-end final pass completed | 62 tests passed*
