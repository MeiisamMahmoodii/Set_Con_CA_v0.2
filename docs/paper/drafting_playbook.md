# Drafting Playbook and Claim Calibration Rules

This is the fastest stable sequence to produce a coherent, reviewer-ready draft without introducing unsupported claims.

## Phase-Ordered Writing Sequence

## Step 1: Freeze metric/statistics definitions
- Finalize exact metric formulas and significance procedures.
- Lock notation and symbols used in method/results.
- Do not write results prose before this is fixed.

## Step 2: Write Sections 5 and 4 first (Method then Notation)
- Draft `Method` with equation-complete objective.
- Draft `Problem Setup and Notation` so all later sections reuse stable symbols.

## Step 3: Write Section 7 (Experimental Setup)
- Add datasets, baselines, split policy, seed policy, statistics protocol.
- Insert table/figure placeholders from `figure_table_map.md`.

## Step 4: Fill Section 8 directly from artifacts only
- One subsection at a time:
  - main results,
  - S sweep,
  - top-k ablation,
  - bridge significance,
  - faithfulness,
  - steering transfer.
- Prohibited: hand-entered numbers not traceable to artifact files.

## Step 5: Write Section 6 (Theory sketches)
- Keep theorem statements concise in main text.
- Move full proofs to Appendix A.
- Ensure assumptions are referenced in experiments where relevant.

## Step 6: Write Sections 3 and 1 last (Related Work and Intro)
- Align novelty statements with what results actually support.
- Avoid overclaiming in introduction.

## Step 7: Write limitations, ethics, reproducibility, conclusion
- Explicitly list unresolved risks, scope boundaries, and reproduction constraints.

## Claim Calibration Rules (Hard Constraints)

1. Use `we prove` only for theorem statements fully proven in appendix.
2. Use `we observe` for empirical findings.
3. Use `in our evaluated setting` for model/data-specific claims.
4. Do not use `universal` unless tested across all declared families with robust evidence.
5. Every number in abstract must appear in main table/figure and artifact file.
6. If a result is partial-run or synthetic-only, mark it as provisional.

## Result Sentence Templates (Safe)

- Theory: `Under Assumptions A1-A3, Proposition 1 shows ...`
- Empirical: `Across seeds {..}, Set-ConCA improves X by Y (95% CI [...]) against baseline Z.`
- Scope-limited: `This effect is observed on Gemma->Llama transfer under the current protocol.`

## Fast Reviewer Red-Team Gate (Run Before Submission)

Answer each with `yes/no`; block submission on any `no`.

1. Can every headline number be traced to script + artifact file?
2. Are baselines strong and fairly tuned?
3. Are test statistics and correction methods explicitly declared?
4. Are failure cases and negative results documented?
5. Is theorem vs empirical wording strictly separated?
6. Could an external group reproduce key tables in under one week?

## Final Quality Pass Checklist

- Citation keys resolve and references are real.
- Notation is consistent across main text and appendix.
- Figure captions include metric definitions and CI/test notes.
- Table footnotes include seeds and statistical method.
- Appendix command blocks reproduce all artifacts.
