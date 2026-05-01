# ConCA Basics

Back to basics: ConCA (Concept Component Analysis) works on **pointwise activations** (`S=1`).

## Core Idea

- Input is one activation vector.
- Model learns sparse concept coordinates.
- Good reconstruction possible.
- But sensitive to local variation/polysemy drift.

## Why It Matters

ConCA is the direct baseline that Set-ConCA extends.  
If we cannot beat or match ConCA in meaningful dimensions, Set-ConCA has no reason to exist.

## Final-Pass Snapshot

| Method | WMT14 mean | OPUS100 mean |
|---|---:|---:|
| ConCA (S=1) | 0.3720 | 0.3725 |
| Set-ConCA | 0.3802 | 0.3688 |

See also: [03_What_Changed_From_ConCA_to_SetConCA](03_What_Changed_From_ConCA_to_SetConCA.md), [15_Baseline_Comparisons](15_Baseline_Comparisons.md)

