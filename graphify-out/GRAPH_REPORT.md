# Graph Report - .  (2026-04-29)

## Corpus Check
- 110 files · ~205,963 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 480 nodes · 874 edges · 39 communities detected
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 183 edges (avg confidence: 0.55)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Init Forward Each|Init Forward Each]]
- [[_COMMUNITY_Exp Baseline Pca|Exp Baseline Pca]]
- [[_COMMUNITY_Train Experiment Run|Train Experiment Run]]
- [[_COMMUNITY_Artifact Neurips Claim|Artifact Neurips Claim]]
- [[_COMMUNITY_Pdf Rationale Paper|Pdf Rationale Paper]]
- [[_COMMUNITY_Anchor Fit Overlap|Anchor Fit Overlap]]
- [[_COMMUNITY_Exp Baseline Family|Exp Baseline Family]]
- [[_COMMUNITY_Plot Generates All|Plot Generates All]]
- [[_COMMUNITY_Baseline Exp Family|Baseline Exp Family]]
- [[_COMMUNITY_Exp Extended Alignment|Exp Extended Alignment]]
- [[_COMMUNITY_Transfer Pointwise Topk|Transfer Pointwise Topk]]
- [[_COMMUNITY_Test Cons Spar|Test Cons Spar]]
- [[_COMMUNITY_Consistency Loss Sparsity|Consistency Loss Sparsity]]
- [[_COMMUNITY_Iter Fren Save|Iter Fren Save]]
- [[_COMMUNITY_Extract Activations Target|Extract Activations Target]]
- [[_COMMUNITY_Test Claim Exp|Test Claim Exp]]
- [[_COMMUNITY_Concept Pointwise Reconstruction|Concept Pointwise Reconstruction]]
- [[_COMMUNITY_Generate Additional Charts|Generate Additional Charts]]
- [[_COMMUNITY_Build Batched Dataset|Build Batched Dataset]]
- [[_COMMUNITY_Transfer Family Bidirectional|Transfer Family Bidirectional]]
- [[_COMMUNITY_Train Build Load|Train Build Load]]
- [[_COMMUNITY_Run Experiments Neurips|Run Experiments Neurips]]
- [[_COMMUNITY_Dataset Build Create|Dataset Build Create]]
- [[_COMMUNITY_Metric Comparison Cluster|Metric Comparison Cluster]]
- [[_COMMUNITY_Make Synthetic Dataset|Make Synthetic Dataset]]
- [[_COMMUNITY_Neighbors Anchor Build|Neighbors Anchor Build]]
- [[_COMMUNITY_Test Corruption Diagnostics|Test Corruption Diagnostics]]
- [[_COMMUNITY_Supervisor Presentation Master|Supervisor Presentation Master]]
- [[_COMMUNITY_Stability Scaling Diminishing|Stability Scaling Diminishing]]
- [[_COMMUNITY_Topk Consistency Mode|Topk Consistency Mode]]
- [[_COMMUNITY_Size Scaling Trend|Size Scaling Trend]]
- [[_COMMUNITY_Low Raw Cka|Low Raw Cka]]
- [[_COMMUNITY_Lower Rank Information|Lower Rank Information]]
- [[_COMMUNITY_Compile Paper|Compile Paper]]
- [[_COMMUNITY_Generate Plots|Generate Plots]]
- [[_COMMUNITY_Init|Init]]
- [[_COMMUNITY_Init|Init]]
- [[_COMMUNITY_Init|Init]]
- [[_COMMUNITY_Init|Init]]

## God Nodes (most connected - your core abstractions)
1. `SetConCA` - 58 edges
2. `AttentionAggregator` - 50 edges
3. `RepresentationSetDataset` - 22 edges
4. `evaluate_pair()` - 21 edges
5. `DualDecoder` - 19 edges
6. `ElementEncoder` - 19 edges
7. `TestLosses` - 19 edges
8. `load_data()` - 18 edges
9. `SetAggregator` - 18 edges
10. `TestFullModel` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Initial Report PDF` --semantically_similar_to--> `Set-ConCA Method`  [INFERRED] [semantically similar]
  initial_report.pdf → results/REPORT.md
- `Summery PDF Report` --references--> `Set-ConCA Method`  [AMBIGUOUS]
  summery.pdf → results/REPORT.md
- `Set-ConCA Technical Presentation` --semantically_similar_to--> `Set-ConCA`  [INFERRED] [semantically similar]
  docs/PRESENTATION.md → README.md
- `Three-Stage Concept Transfer Pipeline` --conceptually_related_to--> `Set-ConCA Framework`  [INFERRED]
  docs/paper/artifacts/pipeline.png → results/REPORT.pdf
- `Semantic Denoising Produces Separated Concept Clusters` --conceptually_related_to--> `Set-ConCA Framework`  [INFERRED]
  docs/paper/artifacts/manifold.png → results/REPORT.pdf

## Hyperedges (group relationships)
- **Set-ConCA Claim Governance Bundle** — claim_traceability_partial_alignment, drafting_playbook_claim_calibration, figure_table_map_artifact_binding, submission_ready_outline_artifact_backed_claims, test_audit_scientific_claim_gaps [EXTRACTED 0.93]
- **ConCA to Set-ConCA Theory-System Bundle** — conca_math_conca_framework, set_conca_math_set_conca, readme_setconca, summery_procrustes_bridge [INFERRED 0.81]
- **Core Set-ConCA Experimental Claims** — concept_cross_family_transfer, concept_interventional_steering, concept_linear_bridge_alignment, concept_topk_sparsity [EXTRACTED 0.90]
- **Set-ConCA Method Design Bundle** — concept_dual_decoder, concept_topk_sparsity, concept_subset_consistency_loss, rationale_sets_capture_invariance [EXTRACTED 0.88]
- **Supervisor Communication Pack** — master_supervisor_presentation_storyboard, presentation_plan_speaker_guide, supervisor_briefing_postmortem, executive_summary_verified_rerun [INFERRED 0.84]
- **Set Training Tradeoff to Transfer Gain** — fig01_set_vs_pointwise_tradeoff, fig04_cross_family_alignment_bars, report_cross_family_transfer_646 [EXTRACTED 0.93]
- **TopK Dominance Across Ablations** — ablation_consistency_redundancy, fig09_consistency_ablation_bars, report_topk_dominant_bias [EXTRACTED 0.95]
- **Set Size and Stability Emergence** — s_sweep_stability_knee, fig02_s_scaling_diminishing_returns, figure2_universal_stability_emergence [INFERRED 0.84]
- **Linear Bridge and Cross-Model Geometry Evidence** — paper_draft_platonic_linear_geometry, bridge_bars_capacity_asymmetry, fig04_cross_family_alignment_bars [INFERRED 0.86]
- **Set-ConCA Visual Narrative** — pipeline_three_stage_concept_transfer, manifold_semantic_denoising_clusters, report_setconca_framework [INFERRED 0.80]
- **Set-ConCA Empirical Evidence Suite** — fig1_set_vs_pointwise_setconca_s8, fig4_cross_family_setconca_transfer, fig6_sota_comparison_setconca_best_balance, fig16_topk_transfer_pointwise_vs_set [INFERRED 0.88]
- **Architecture to Experiment Validation Loop** — fig10_architecture_system_map, fig3_aggregator_ablation_mean_pool, fig15_soft_consistency_soft_sparsity_benefit, fig10_corruption_test_topk_robustness [EXTRACTED 0.93]
- **Representation Bridge Analysis Family** — fig11_layer_sweep_lower_rank_better, fig12_nonlinear_bridge_linear_procrustes, fig14_pca32_transfer_pca32_distillation, fig5_intra_family_overlap_with_bridge [INFERRED 0.83]

## Communities

### Community 0 - "Init Forward Each"
Cohesion: 0.04
Nodes (55): AttentionAggregator, Learned attention-based aggregation of set elements.     Uses a learnable query, SetAggregator, Dataset, Each item is a set of hidden states of shape (set_size, hidden_dim).     Sets re, RepresentationSetDataset, DualDecoder, Shared + Residual decoder.     f_hat(x_i) = W_shared * z_hat + W_residual * u_i (+47 more)

### Community 1 - "Exp Baseline Pca"
Cohesion: 0.19
Nodes (32): baseline_pca(), baseline_pca_threshold(), baseline_random(), baseline_sae_l1(), baseline_sae_topk(), ci95(), cka(), concept_disentanglement() (+24 more)

### Community 2 - "Train Experiment Run"
Cohesion: 0.08
Nodes (24): apply_steering(), cka(), evaluate_transfer(), Centered Kernel Alignment over representations., Computes overlap fraction between Top-K elements of two vectors of activations., x: target model residual stream activation     z_concept: source model set-conca, topk_overlap(), Experiment 1: Set vs Pointwise     data shape: (N_sets, S, D) (+16 more)

### Community 3 - "Artifact Neurips Claim"
Cohesion: 0.08
Nodes (28): Multilingual Model Deferrals, NeurIPS Artifact Requirements, Claim-Evidence Partial Alignment, Concept Component Analysis (ConCA), Latent Variable Model for Text, Sparse Autoencoders (SAE), Claim Calibration Rules, Figure-Artifact Binding Spec (+20 more)

### Community 4 - "Pdf Rationale Paper"
Cohesion: 0.09
Nodes (26): ConCA Foundational Paper, Cross-Family Concept Transfer, Dual Decoder Shared Residual, Interventional Concept Steering, Linear Procrustes Bridge Alignment, Platonic Representation Hypothesis, Subset Consistency Loss, TopK Hard Sparsity (+18 more)

### Community 5 - "Anchor Fit Overlap"
Cohesion: 0.22
Nodes (24): _anchor_mean_codes(), _available_pair_matrix(), _bridge_overlap(), _cca_family_overlap(), _contrastive_linear_overlap(), evaluate_pair(), _fit_conca_anchor(), _fit_gated_sae_anchor() (+16 more)

### Community 6 - "Exp Baseline Family"
Cohesion: 0.24
Nodes (20): baseline_ica(), baseline_pca(), baseline_random(), baseline_sae(), cka(), exp1_set_vs_pointwise(), exp2_s_scaling(), exp3_aggregator_ablation() (+12 more)

### Community 7 - "Plot Generates All"
Cohesion: 0.22
Nodes (16): fig01(), fig02(), fig03(), fig04(), fig05(), fig06(), fig07(), fig08() (+8 more)

### Community 8 - "Baseline Exp Family"
Cohesion: 0.29
Nodes (15): baseline_ica(), baseline_pca(), baseline_random(), baseline_sae(), cka(), exp4_cross_family(), exp5_intra_family(), exp6_sota_comparison() (+7 more)

### Community 9 - "Exp Extended Alignment"
Cohesion: 0.24
Nodes (15): cca_overlap(), exp_asymmetry_diagnostics(), exp_cross_language_if_available(), exp_layerwise_alignment_and_steering(), exp_sota_extensions(), _flatten_first_paraphrase(), ica_codes(), _log() (+7 more)

### Community 10 - "Transfer Pointwise Topk"
Cohesion: 0.16
Nodes (14): Set-ConCA System Architecture Map, TopK Transfer Robustness Under Corruption, Linear Procrustes Bridge, Nonlinear MLP Bridge, Soft-Sparsity Consistency Benefit, TopK Pointwise vs Set Transfer Comparison, ConCA / Pointwise (S=1) Method, Set-ConCA (S=8) Method (+6 more)

### Community 11 - "Test Cons Spar"
Cohesion: 0.21
Nodes (1): TestLosses

### Community 12 - "Consistency Loss Sparsity"
Cohesion: 0.2
Nodes (5): consistency_loss(), Subset consistency regularization.     Splits x into two disjoint halves, runs b, probability_domain_l1(), L1 sparsity on probability-mapped latents (paper Sec. 4.4).      Input: u_bar =, sparsity_loss()

### Community 13 - "Iter Fren Save"
Cohesion: 0.38
Nodes (6): BenchmarkRecord, build_records(), _is_good_pair(), iter_europarl_fren(), iter_wmt14_fren(), normalize_text()

### Community 14 - "Extract Activations Target"
Cohesion: 0.27
Nodes (3): ActivationExtractor, extract_real_activations(), Extracts activations using the exact target LLM layer and groups them into

### Community 15 - "Test Claim Exp"
Cohesion: 0.29
Nodes (5): _load_results(), test_CLAIM_01_exp9_framing_matches_transfer_direction(), test_CLAIM_02_exp10_near_chance_only_if_numeric_supports_it(), test_CLAIM_03_exp14_improvement_text_matches_values(), test_CLAIM_04_exp16_primary_driver_statement_matches_diff()

### Community 16 - "Concept Pointwise Reconstruction"
Cohesion: 0.25
Nodes (8): Set vs Pointwise Reconstruction-Stability Tradeoff, Attention vs Mean Aggregator Ablation, Sparse SOTA Comparison Panel, Interventional Steering and Weak-to-Strong Curves, Training Convergence Plateau Near Epoch 50, Semantic Denoising Produces Separated Concept Clusters, Three-Stage Concept Transfer Pipeline, Set-ConCA Framework

### Community 17 - "Generate Additional Charts"
Cohesion: 0.48
Nodes (6): arrow(), draw_box(), fig10_architecture(), fig9_evolution(), generate_additional_charts.py Produces:   fig9_evolution.png   — ConCA → Set-Con, Draw a rounded rectangle with centred multiline text.

### Community 18 - "Build Batched Dataset"
Cohesion: 0.67
Nodes (5): _batched(), build_dataset(), extract_for_model(), _log(), save_model_tensor()

### Community 19 - "Transfer Family Bidirectional"
Cohesion: 0.33
Nodes (6): Capacity Asymmetry in Bidirectional Transfer, Mid-Layer Faithfulness Peak, Cross-Family Alignment Bidirectional Bars, Intra-Family Transfer Heatmap, Approximate Linear Platonic Geometry, Cross-Family Transfer 64.6%

### Community 20 - "Train Build Load"
Cohesion: 0.7
Nodes (4): build_model(), load_data(), parse_args(), train()

### Community 21 - "Run Experiments Neurips"
Cohesion: 0.6
Nodes (3): cka(), run_all_experiments(), topk_overlap()

### Community 22 - "Dataset Build Create"
Cohesion: 0.6
Nodes (3): build_dataset(), create_anchor(), generate_variations()

### Community 23 - "Metric Comparison Cluster"
Cohesion: 0.5
Nodes (5): Cluster NMI Metric, Linear Probe Accuracy Metric, Capability Comparison Matrix, Set-ConCA Best Trade-off Profile, Method Comparison Radar

### Community 24 - "Make Synthetic Dataset"
Cohesion: 0.5
Nodes (2): make_synthetic_dataset(), Synthetic Gaussian data for unit tests.

### Community 25 - "Neighbors Anchor Build"
Cohesion: 0.67
Nodes (2): build_anchor_neighbors(), Given a list of items:     {       "anchor_id": "...",       "vector": [np.array

### Community 26 - "Test Corruption Diagnostics"
Cohesion: 1.0
Nodes (2): test_corruption_diagnostics(), topk_overlap()

### Community 27 - "Supervisor Presentation Master"
Cohesion: 0.67
Nodes (3): Master Supervisor Presentation, Presentation Plan Speaker Guide, Supervisor Briefing Postmortem

### Community 28 - "Stability Scaling Diminishing"
Cohesion: 0.67
Nodes (3): S-Scaling Diminishing Returns Curve, Universal Stability Emergence Across Model Families, Set Size Stability Knee Around S=8

### Community 29 - "Topk Consistency Mode"
Cohesion: 0.67
Nodes (3): Consistency Loss Redundancy in TopK Mode, Consistency Ablation Bars in TopK Mode, TopK Dominant Inductive Bias

### Community 30 - "Size Scaling Trend"
Cohesion: 1.0
Nodes (2): Set Size Scaling Trend, Stability Improves with Larger S

### Community 31 - "Low Raw Cka"
Cohesion: 1.0
Nodes (2): Low Raw CKA Across Gemma Sizes, High Transfer Overlap with Bridge

### Community 32 - "Lower Rank Information"
Cohesion: 1.0
Nodes (2): Lower-Rank Information Gives Better Transfer, PCA-32 Input Distillation Result

### Community 33 - "Compile Paper"
Cohesion: 1.0
Nodes (0): 

### Community 34 - "Generate Plots"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Init"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Init"
Cohesion: 1.0
Nodes (0): 

## Ambiguous Edges - Review These
- `Synthetic Reproduction Artifacts` → `NeurIPS Suite Bridge Overlap`  [AMBIGUOUS]
  docs/paper/reproduction_report.md · relation: conceptually_related_to
- `Set-ConCA Method` → `Summery PDF Report`  [AMBIGUOUS]
  summery.pdf · relation: references
- `Approximate Linear Platonic Geometry` → `Mid-Layer Faithfulness Peak`  [AMBIGUOUS]
  docs/paper/artifacts/faith_layer.pdf · relation: conceptually_related_to
- `Set-ConCA Best Trade-off Profile` → `Cluster NMI Metric`  [AMBIGUOUS]
  results/figures/fig13_interpretability.png · relation: conceptually_related_to

## Knowledge Gaps
- **65 isolated node(s):** `generate_additional_charts.py Produces:   fig9_evolution.png   — ConCA → Set-Con`, `Draw a rounded rectangle with centred multiline text.`, `plot_results.py =============== Generates all publication-quality figures for Se`, `Does performance vary across LLM layers? (Simulated via subsets of D dims.)`, `run_extended_alignment.py ========================= Extended analyses for:` (+60 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Size Scaling Trend`** (2 nodes): `Set Size Scaling Trend`, `Stability Improves with Larger S`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Low Raw Cka`** (2 nodes): `Low Raw CKA Across Gemma Sizes`, `High Transfer Overlap with Bridge`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Lower Rank Information`** (2 nodes): `Lower-Rank Information Gives Better Transfer`, `PCA-32 Input Distillation Result`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Compile Paper`** (1 nodes): `compile_paper.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generate Plots`** (1 nodes): `generate_plots.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Synthetic Reproduction Artifacts` and `NeurIPS Suite Bridge Overlap`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Set-ConCA Method` and `Summery PDF Report`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Approximate Linear Platonic Geometry` and `Mid-Layer Faithfulness Peak`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Set-ConCA Best Trade-off Profile` and `Cluster NMI Metric`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `SetConCA` connect `Init Forward Each` to `Train Experiment Run`, `Test Cons Spar`, `Consistency Loss Sparsity`?**
  _High betweenness centrality (0.156) - this node is a cross-community bridge._
- **Why does `AttentionAggregator` connect `Init Forward Each` to `Test Cons Spar`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `run_evaluation_resume.py Resumes from saved partial results, runs EXP4-7, and me` connect `Init Forward Each` to `Baseline Exp Family`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._