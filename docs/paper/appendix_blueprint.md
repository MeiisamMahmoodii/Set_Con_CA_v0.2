# Appendix Blueprint (Deep Version)

Use this as the exact appendix contract so the main paper remains concise while the submission is fully auditable.

## Appendix A: Mathematical Development

### A.1 Assumption ledger
- List each assumption with identifier and where it is used.
- Include realistic interpretation and practical failure mode.

### A.2 Lemmas
- Technical lemmas used in propositions.
- Keep one lemma per subsection for readability.

### A.3 Proposition 1 proof (set aggregation consistency)
- Full derivation with variance scaling and constants.
- Note boundary cases for small set size.

### A.4 Proposition 2 proof (posterior aggregation interpretation)
- Full derivation from pointwise proxy to set-level estimator.
- Clarify approximation terms.

### A.5 Proposition 3 proof (transplantability condition)
- Conditions for bridge identifiability.
- Show null-relative expected overlap argument.

### A.6 Assumption stress tests
- Counterexamples where assumptions break.
- Expected empirical symptom in experiments.

## Appendix B: Data and Experimental Protocol

### B.1 Dataset sources and curation
- Sources, filtering rules, and exclusions.

### B.2 Anchor synchronization protocol
- Exact anchor matching criteria.
- Tie-break and fallback rules.

### B.3 Split policy and leakage prevention
- Train/val/test definitions.
- Leakage checks and invariants.

### B.4 Seed policy
- Seed list and deterministic flags.

## Appendix C: Implementation and Training Details

### C.1 Architecture config tables
- Hidden dims, concept dims, set size, k values.

### C.2 Optimization details
- Optimizer, lr schedule, clipping, stopping criteria.

### C.3 Runtime settings
- Batch sizes, precision mode, gradient accumulation.

### C.4 Exact command lines
- Reproducible command blocks for every table/figure artifact.

## Appendix D: Extended Ablations

### D.1 Full S sweep table
- Include all S values and all key metrics.

### D.2 Full k sweep table
- Include performance and stability tradeoffs.

### D.3 Aggregator variant ablation
- Mean vs attention comparisons.

### D.4 Norm-alignment ablation
- On/off and calibration variants.

## Appendix E: Statistical Procedures

### E.1 Bootstrap protocol
- Resampling unit, number of resamples, CI method.

### E.2 Permutation tests
- Null construction and permutation count.

### E.3 Multiple comparisons
- Correction strategy and rationale.

### E.4 Effect size reporting
- Define effect size formulas and interpretation thresholds.

## Appendix F: Qualitative and Failure Analysis

### F.1 Concept examples
- Human-readable examples for high-confidence concepts.

### F.2 Steering examples
- Before/after behavior snapshots.

### F.3 Failure cases
- Cases where transfer fails, with diagnosis.

### F.4 Negative results
- Explicitly include non-winning settings.

## Appendix G: Compute and Resource Accounting

### G.1 Hardware inventory
- GPU/CPU model, memory, software versions.

### G.2 Runtime and memory table
- Per experiment class.

### G.3 Cost/carbon reporting
- Methodology and estimates.

## Appendix H: Reproducibility Checklist and Artifact Manifest

### H.1 Checklist
- Data access, scripts, seeds, metrics, tests.

### H.2 Artifact manifest table
- Columns:
  - section/table/figure id
  - script path
  - command
  - output file
  - checksum (optional)

### H.3 Reproducible subset (<1 week)
- Define minimal reproducible package and expected outputs.
