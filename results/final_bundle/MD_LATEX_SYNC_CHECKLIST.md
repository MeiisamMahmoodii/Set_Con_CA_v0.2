# Markdown <-> LaTeX Sync Checklist

## Primary files
- Markdown primary: `results/final_bundle/Submission_Master_Paper.md`
- LaTeX primary: `docs/paper/setconca_neurips.tex`

## Locked headline claims (must match)
- EXP4 transfer (Gemma-3 4B -> LLaMA-3 8B): `69.5% +/- 0.6pp`
- EXP7 steering gain at alpha=10: `+9.8pp`
- EXP12 linear vs MLP: `69.3% vs 64.2%`
- EXP16 pointwise vs set overlap: `78.4% vs 69.5%`
- Multilingual Set-ConCA means: `0.3802 / 0.3688`

## Required narrative consistency
- Include strong + mixed + negative findings in both versions.
- Use conservative claim framing ("competitive, not dominant").
- Explicitly avoid prohibited claims:
  - universal baseline dominance
  - strict consistency necessity in current TopK mode
  - corruption-collapse proof claim

## Verification references
- Canonical report: `results/REPORT.md`
- Claim ledger: `results/final_bundle/ClaimLedger_vFinal.json`
- Report checker: `scripts/verify_reports_against_artifacts.py`
