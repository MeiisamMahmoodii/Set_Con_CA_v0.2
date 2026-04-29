# Multilingual: WMT14 + OPUS100

This section isolates multilingual outcomes and model coverage.

## Datasets Used

- `wmt14_fr_en`
- `opus100_multi_en` (English-inclusive multilingual follow-up)

## Models Included (final pass)

- Gemma-2-2B
- Llama-3.2-1B-Instruct
- Llama-3.2-3B-Instruct
- Mistral-7B-Instruct-v0.3
- Phi-3.5-mini-instruct
- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct

## Comparative Means

| Method | WMT14 | OPUS100 |
|---|---:|---:|
| Set-ConCA | 0.3802 | 0.3688 |
| ConCA (S=1) | 0.3720 | 0.3725 |
| CCA | 0.3433 | 0.3196 |
| PWCCA | 0.4839 | 0.3864 |

## Notes

- Pipeline is now complete and reproducible for these two benchmarks.
- Positioning is "competitive", not "dominant on raw overlap".

See:
- [[12_Final_Results_Snapshot]]
- [[11_Findings_Failures_and_Limits]]

