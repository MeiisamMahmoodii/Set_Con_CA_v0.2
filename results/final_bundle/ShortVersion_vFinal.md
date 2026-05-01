# Set-ConCA Short Version (vFinal)

## Core status
- Tests pass: `62 passed`
- Canonical metrics: `results/results_v2.json`
- Cross-family transfer (Gemma-3 4B -> LLaMA-3 8B): **69.5% +/- 0.6pp** (chance 25%)
- Steering gain at alpha=10: **+9.8pp** (weak-to-strong: **+10.7pp**)
- Linear bridge vs MLP: **69.3% vs 64.2%**

## Critical caveats
- Pointwise TopK outperforms Set-ConCA on raw overlap: **78.4% vs 69.5%**
- Consistency effect is small in current TopK setup (**+0.1pp**)
- Corruption test does not support collapse-to-chance claim

## Multilingual snapshot
- Pipeline operational on WMT14 FR-EN and OPUS100 multi-EN
- Set-ConCA matrix means: **0.3802 / 0.3688**
- Safe framing: competitive, not dominant

## Read next
- Full report: `results/final_bundle/FullReport_vFinal.md`
- PhD cheat sheet: `results/final_bundle/CheatSheet_vFinal.md`
- Meeting deep-dive: `results/final_bundle/Supervisor_Meeting_Paper.md`
