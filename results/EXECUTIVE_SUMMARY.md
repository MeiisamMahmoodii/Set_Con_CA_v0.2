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
* **Cross-language EN/FR:** now has a real **WMT14 fr-en** benchmark path with completed tensors for **Qwen-3B**, **Qwen-7B**, **Mistral-7B**, and **Gemma-2-2B**.
* **WMT14 multilingual result:** Set-ConCA averages **0.3187** raw overlap over the completed 8-pair matrix; this is competitive but **not** a win over several dense/pointwise alignment references on the current 128-anchor run.

## Current Positioning
* **Best framing:** Set-ConCA is a credible set-based sparse concept method with strong cross-family transfer and steering evidence, plus a now-working multilingual benchmark path.
* **Unsafe framing now:** claiming consistency is essential, corruption proves semantic dependence, PCA-32 improves transfer, or Set-ConCA beats all strong baselines on raw WMT14 EN/FR overlap.

---
*Verified on RTX 3090 GPU | 2,048 anchors | 5 seeds | 60 tests passed*
