# Project Story: From Zero to Full Technical (vFinal)

## Who this document is for

This document is for someone who is **not technical** but wants to understand this project from beginning to end, including:

- what problem we tried to solve,
- what we built and why,
- what results were good vs weak,
- what changed over time,
- and why the final claims are now more trustworthy.

For each major part, the structure is always:

1. Simple version (plain language)
2. Technical version (how it actually works)
3. Results (what we measured)
4. History of changes (what we used to claim vs what we now claim)
5. Transition to next part

---

## Part 1 - Why this project exists

### 1A) Simple version

Imagine you ask several people to describe the same idea using different words.  
You want a system that captures the **shared meaning**, not the specific wording.

Most older methods look at one sentence at a time. That often mixes meaning with style and wording.

This project asks: can we learn concept representations that are more stable by looking at **sets** of equivalent examples instead of single examples?

### 1B) Technical version

The project extends pointwise concept extraction (ConCA, `S=1`) into set-based concept extraction (Set-ConCA, `S>1`), where each training item is a paraphrase set.  
The core hypothesis is that a set-level bottleneck encourages semantic invariance and better cross-model transfer behavior.

### 1C) Results snapshot

Final rerun status:

- Full test suite: **62 passed**
- Canonical report: `results/REPORT.md`
- Canonical metrics: `results/results_v2.json`
- Final multilingual matrices: WMT14 + OPUS100 complete, **7 models / 26 directed pairs** each

### 1D) History of changes

Earlier project framing was broader and more optimistic.  
After a full rerun + contradiction checks, claims were narrowed to what the data supports.

### 1E) Transition

Now that the motivation is clear, we start with the baseline method that existed before Set-ConCA.

---

## Part 2 - The baseline (ConCA) before Set-ConCA

### 2A) Simple version

ConCA is like taking one photo at a time and trying to infer the concept from each photo alone.  
It can work, but it can over-focus on surface details in that single photo.

### 2B) Technical version

ConCA operates pointwise (`S=1`): each activation vector is independently encoded into a sparse concept code and decoded.  
No set aggregation is used, so there is no explicit multi-view consistency pressure.

### 2C) Results

In final multilingual matrix means:

- ConCA(S=1): **0.3720** (WMT14), **0.3725** (OPUS100)
- Set-ConCA: **0.3802** (WMT14), **0.3688** (OPUS100)

Interpretation: Set-ConCA is close/competitive with ConCA(S=1), but not uniformly better on every dataset/metric.

### 2D) History of changes

Earlier storytelling sometimes treated ConCA as clearly weaker everywhere.  
Final evidence shows a mixed picture: ConCA remains strong in some comparisons.

### 2E) Transition

Next: what exactly changed when moving from ConCA to Set-ConCA.

---

## Part 3 - What Set-ConCA changed

### 3A) Simple version

Instead of asking the model to understand one sentence in isolation, we ask it to understand a **small group of equivalent sentences** together.  
This is intended to make the learned concept more about meaning and less about wording.

### 3B) Technical version

Main deltas from ConCA to Set-ConCA:

- Input unit: point -> set (`S>=1`)
- Add set aggregator (mean or attention)
- Sparse concept bottleneck retained
- Add consistency-oriented behavior in training design
- Evaluate more aggressively on transfer, steering, and multilingual matrix benchmarks

### 3C) Results

Architecture-level findings:

- Set-size scaling (EXP2): more paraphrases improve reconstruction; stability not strictly monotonic
- Aggregator ablation (EXP3): in final rerun, **attention** outperforms mean on tracked metrics

### 3D) History of changes

Earlier internal narratives often promoted mean aggregation as the empirical winner.  
Final rerun corrected this: mean remains simpler, but attention wins the measured comparison in this run.

### 3E) Transition

Now we walk through the full experiment program in chronological order.

---

## Part 4 - Experiments EXP1 to EXP16 (the full story)

## EXP1: Set vs Pointwise

### Simple
Set training is harder than single-point training, so it may reconstruct slightly worse but potentially transfer better.

### Technical
Compare Set-ConCA (`S=8`) vs pointwise (`S=1`) on MSE and stability.

### Results

- Set-ConCA MSE: **0.1735**
- Pointwise MSE: **0.1061**
- Stability: near tie

### History
This remained stable: set method is not the MSE winner in the local reconstruction task.

### Transition
If set training is harder, does changing set size help?

## EXP2: Set-size scaling

### Simple
Do more paraphrases help?

### Technical
Sweep `S={1,3,8,16,32}`.

### Results

- MSE improves as `S` increases
- Stability is flatter than expected
- `S=8` remains practical default

### History
Old claim: clear stability knee around `S=8`.  
Final claim: practical compromise yes, sharp knee no.

### Transition
If sets matter, how should we combine set members?

## EXP3: Aggregator ablation

### Simple
How do we combine multiple paraphrases into one concept signal?

### Technical
Mean pooling vs attention aggregation.

### Results

- Attention: better MSE and stability in this rerun

### History
Old claim: mean better stability.  
Final claim: attention empirically better; mean is simpler.

### Transition
Core question next: does transfer across model families work?

## EXP4: Cross-family transfer (headline)

### Simple
Can concepts learned in one model be recognized in a very different model?

### Technical
Gemma-3 4B -> LLaMA-3 8B via linear bridge; evaluate top-k overlap transfer.

### Results

- Transfer: **69.5% +/- 0.6pp**
- Chance level: **25%**
- Reverse direction: **59.6% +/- 3.5pp**

### History
Old headline values were lower and noisier.  
Final rerun confirms strong, reproducible cross-family transfer.

### Transition
But is cross-family always better than within-family?

## EXP5: Intra-family transfer

### Simple
Do models from the same family transfer concepts more easily?

### Technical
Transfer matrix across Gemma variants.

### Results

- 1B -> 4B: **64.9%**
- 4B -> 1B: **69.1%**
- 4B -> 9B: **54.4%**
- 9B -> 4B: **64.1%**

### History
Old simple story ("cross-family always > intra-family") is false.  
Final story: asymmetry is nuanced; capacity mismatch and training recipe both matter.

### Transition
How does Set-ConCA compare to strong baselines overall?

## EXP6: SOTA-style baseline comparison

### Simple
Is Set-ConCA actually better than alternatives?

### Technical
Compare sparse/dense baselines on MSE, stability, sparsity, and notes on transfer context.

### Results

- Set-ConCA MSE: **0.1735**
- SAE-TopK MSE: **0.1868**
- Set-ConCA has strong sparse-method reconstruction trade-off

### History
Some old framing overstated transfer dominance vs SAE-TopK.  
Final framing: strong trade-off and transfer evidence, but no universal dominance claim.

### Transition
Can learned concepts cause predictable behavior changes?

## EXP7: Causal steering

### Simple
If a concept is real, pushing that concept should change model behavior in predictable ways.

### Technical
Intervene along learned concept directions; measure similarity change vs intervention strength (`alpha`), including controls.

### Results

- Set-ConCA gain at `alpha=10`: **+9.8pp**
- Weak-to-strong gain: **+10.7pp**
- Random control degrades sharply

### History
This result became stronger in the verified rerun and is one of the project's strongest supports.

### Transition
Before trusting this, we verify training stability.

## EXP8: Convergence

### Simple
Did training behave consistently, or was it unstable luck?

### Technical
Inspect convergence curves over training schedule.

### Results

- Stable convergence with current 80-epoch setup

### History
Convergence remains stable in final pass.

### Transition
Now test whether consistency loss truly drives transfer.

## EXP9: Consistency ablation

### Simple
If we remove consistency loss, does performance collapse?

### Technical
Compare `beta=0.01` vs `beta=0` in TopK mode.

### Results

- Full: **69.5%**
- No consistency: **69.4%**
- Difference: about **+0.1pp**

### History
Old stronger claim ("consistency essential") was corrected.  
Final claim: in TopK mode, consistency is not a dominant transfer driver.

### Transition
Next stress test: corrupt set information and observe effect.

## EXP10: Corruption test

### Simple
If set semantics are broken, does transfer crash to chance?

### Technical
Corrupt set contents at 0/50/100% and evaluate transfer.

### Results

- 0%: **69.3%**
- 50%: **70.1%**
- 100%: **69.2%**

### History
Old claim: collapse-to-chance under corruption.  
Final claim: no collapse under this corruption protocol; treat as negative/neutral for semantic-collapse hypothesis.

### Transition
Now test low-rank information hypotheses.

## EXP11 and EXP14: PCA-rank proxy vs explicit PCA-distilled input

### Simple
Do compressed representations help transfer?

### Technical

- EXP11: proxy analysis with PCA-rank sweeps
- EXP14: explicit PCA-32 distilled-input transfer test

### Results

- EXP11 rank-32 proxy peak: **72.3%**
- EXP14 explicit PCA-32 transfer: **31.4% +/- 1.3pp**

### History
Old broad claim: "PCA-32 helps transfer."  
Final correction: these are different interventions; explicit PCA-32 hurts in EXP14.

### Transition
Next: test whether nonlinearity in bridge helps.

## EXP12: Linear vs nonlinear bridge

### Simple
Do we need a complex nonlinear map between model concept spaces?

### Technical
Compare linear bridge to MLP bridge.

### Results

- Linear: **69.3%**
- MLP: **64.2%**

### History
Older framing implied near-tie or slight nonlinear gain.  
Final rerun shows linear is better.

### Transition
What about single-model interpretability quality?

## EXP13: Interpretability proxy metrics

### Simple
Are learned concepts easy to interpret inside one model?

### Technical
Compare NMI/probe metrics across methods.

### Results

- Set-ConCA: NMI **0.860**, probe **98.5%**
- SAE-L1: NMI **0.882**
- PCA: NMI **0.924**

### History
Final framing: Set-ConCA is competitive, but no clean single-model interpretability dominance.

### Transition
Final direct transfer showdown: pointwise TopK vs set method.

## EXP16: Pointwise TopK vs Set-ConCA transfer

### Simple
In raw overlap transfer alone, who wins?

### Technical
Direct comparison under current setup.

### Results

- Pointwise SAE-TopK: **78.4% +/- 4.6pp**
- Set-ConCA: **69.5% +/- 0.6pp**

### History
Old broad dominance claims were removed.  
Final claim: Set-ConCA is credible/competitive, but not top on this raw-overlap metric.

### Transition
Next we expand beyond core EXPs into extended diagnostics and multilingual scaling.

---

## Part 5 - Extended diagnostics and multilingual expansion

### 5A) Simple version

After proving the main pipeline works, the team tested broader scenarios:

- more baseline families,
- layer/depth proxy behavior,
- multilingual benchmark matrices.

### 5B) Technical version

Extended diagnostics source: `results/extended_alignment_results.json`  
Multilingual matrices:

- `results/benchmark_matrix_wmt14_fr_en.json`
- `results/benchmark_matrix_opus100_multi_en.json`

### 5C) Results

Selected extended baseline overlaps:

- Procrustes: **0.7302**
- Ridge: **0.7242**
- CCA: **0.7300**
- NMF: **0.8348**
- ICA: **0.1307**

Layer/depth proxy findings:

- Best pseudo-layer pair: **early -> mid = 0.7413**
- Relative-depth 60% map: **mid -> mid = 0.7405**

Multilingual completion:

- Datasets: WMT14 fr-en and OPUS100 multi-en
- Coverage: **7 models / 26 directed pairs** each
- Set-ConCA mean: **0.3802** (WMT14), **0.3688** (OPUS100)
- ConCA(S=1): **0.3720 / 0.3725**

### 5D) History of changes

Earlier documentation treated multilingual benchmarking as partially blocked.  
Final pass completed both canonical matrix artifacts and baseline comparisons.

### 5E) Transition

Now we explain why trust in this project improved after the audit and rerun process.

---

## Part 6 - Why trust improved (audit, corrections, verification)

### 6A) Simple version

At one point, parts of the story were too optimistic.  
The team then performed a cleanup: reran experiments, fixed contradictions, and removed claims the data did not support.

### 6B) Technical version

Trust upgrades included:

- full rerun with canonical artifacts,
- claim-level contradiction checks,
- validation gate tests,
- regenerated figures and updated framing,
- explicit source-of-truth rules and claim ledger.

### 6C) Results

- `pytest`: **62 passed**
- canonical narrative synchronized with `results/*.json`
- claim-evidence mapping formalized in `ClaimLedger_vFinal.md`

### 6D) History of changes (key examples)

1. **Consistency claim**
   - Old: consistency is essential
   - New: in TopK mode, effect is tiny (~+0.1pp)

2. **Corruption claim**
   - Old: corruption collapses transfer to chance
   - New: no collapse under tested corruption protocol

3. **PCA-32 claim**
   - Old: PCA-32 helps transfer
   - New: EXP11 proxy and EXP14 explicit intervention differ; explicit PCA-32 hurts

4. **Dominance claim**
   - Old: broad baseline dominance
   - New: competitive framing; pointwise TopK leads raw overlap in current setup

### 6E) Transition

With trust basis established, we can state the final honest position clearly.

---

## Part 7 - Final honest position (what is strong, mixed, unsafe)

### 7A) Simple version

The project succeeded in important ways, but not in every way.

### 7B) Technical version

Best-defended core:

- robust cross-family transfer signal,
- strong causal steering,
- linear bridge sufficiency,
- competitive sparse reconstruction/transfer trade-off.

### 7C) Results-backed summary

Strong:

- EXP4: **69.5% +/- 0.6pp** vs **25% chance**
- EXP7: **+9.8pp** steering gain, weak-to-strong **+10.7pp**
- EXP12: linear **69.3%** > MLP **64.2%**

Mixed/weak:

- EXP16 raw overlap: Set-ConCA < SAE-TopK
- EXP9 consistency in TopK: small impact
- EXP10 corruption: no semantic-collapse support
- EXP13 single-model interpretability: competitive, not clearly best

Unsafe claims:

- universal raw-overlap dominance,
- strict consistency necessity in TopK,
- corruption-collapse proof,
- universal PCA-32 improvement.

### 7D) History of changes

This section reflects the post-audit narrowing from broad superiority language to evidence-calibrated claims.

### 7E) Transition

Final part: how a reader can navigate all project documents at different depth levels.

---

## Part 8 - How to read and verify everything

### 8A) 30-minute read path (decision-maker)

1. `results/EXECUTIVE_SUMMARY.md`
2. `results/REPORT.md` (Main Takeaways + Honest Bottom Line sections)
3. `results/final_bundle/ClaimLedger_vFinal.md`

### 8B) 2-hour read path (deep stakeholder)

1. Full `results/REPORT.md` (EXP1-EXP16 + Extended Diagnostics)
2. `docs/report/narrative/01_...` through `16_...`
3. `results/final_bundle/MasterReport_vFinal.md`
4. `results/final_bundle/Supervisor_Meeting_Paper.md`

### 8C) Full technical verification path

1. Metric artifacts:
   - `results/results_v2.json`
   - `results/extended_alignment_results.json`
   - multilingual matrix json files
2. Report sync rules:
   - `results/final_bundle/SOURCE_OF_TRUTH_RULES.md`
   - `results/final_bundle/DOCUMENTATION_MANIFEST.md`
3. Validation:
   - `tests/test_setconca.py`
   - `tests/test_validation_gates.py`
4. Reproduction support:
   - `docs/paper/reproduction_report.md`
   - `docs/paper/test_audit_report.md`

---

## Appendix A - Glossary for non-technical readers

- **Activation**: internal numerical pattern inside a model when processing text.
- **Concept code**: compressed representation intended to capture meaning.
- **Sparse**: only a small number of concept dimensions are active at once.
- **TopK**: keep only the top K strongest concept activations.
- **Transfer**: whether concept structure learned in one model aligns with another.
- **Steering**: intentionally nudging model behavior using concept directions.
- **Bridge**: mapping from one model's concept space to another.
- **Cross-family**: transfer between different model families (e.g., Gemma -> LLaMA).
- **Intra-family**: transfer within one model family.
- **Chance level**: performance expected by random guessing.

---

## Appendix B - Claim-to-evidence map (human-friendly)

- **C001** Cross-family transfer is strong  
  Evidence: `results/results_v2.json` -> `exp4_cross_family.SetConCA.transfer_g_to_l`

- **C002** Steering gains are meaningful  
  Evidence: `results/results_v2.json` -> `exp7_steering.gain_at_alpha10_4B`

- **C003** Linear bridge is sufficient or better  
  Evidence: `results/results_v2.json` -> `exp12_nonlinear_bridge.summary`

- **C004** Pointwise TopK beats Set-ConCA on raw overlap in current setup  
  Evidence: `results/results_v2.json` -> `exp16_topk_pointwise_vs_set`

- **C005** Consistency loss is not dominant in TopK mode  
  Evidence: `results/results_v2.json` -> `exp9_consistency_ablation`

- **C006** Corruption does not collapse transfer to chance under tested protocol  
  Evidence: `results/results_v2.json` -> `exp10_corruption_test`

- **C007** Multilingual benchmark path is operational (WMT14 + OPUS100)  
  Evidence: matrix artifacts in `results/benchmark_matrix_*`

- **C008** Extended diagnostics include strong Procrustes/NMF-style signals  
  Evidence: `results/extended_alignment_results.json` -> `sota_extensions`

---

## Final plain-language conclusion

If you remember only one message, use this:

Set-ConCA is a serious and credible set-based concept method.  
Its strongest evidence is cross-family transfer and causal steering.  
Its weaker areas are raw-overlap dominance claims and some earlier over-strong interpretations that were corrected after rerun and audit.  
Because those corrections were made explicitly and tied to artifacts, the current version is much more trustworthy.

