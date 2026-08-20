# RSM-25M result

Status: promising internal continual-memory result; **not yet a field-level breakthrough claim**.

## Final architecture

- Slow model: 8-layer causal Transformer, d=512, 8 heads, byte vocab 260.
- Slow parameters: 25,565,696.
- Fast/plastic budget: 133,120.
- Total parameter budget: 25,698,816.
- Fast memory: 16 routed banks × 32 projected features × 260 outputs.
- A new fact updates only its routed fast-weight bank with one local SGD/outer-product update while slow weights stay frozen.

## Final seed-42 held-out result after pretraining + SFT + programmatic preference post-training

| Sequential facts | Dense fast weights | Routed 16×32 |
|---:|---:|---:|
| 1 | 100% | 100% |
| 4 | 32.5% | 90.0% |
| 8 | 31.25% | 82.5% |
| 16 | 24.375% | 66.25% |

Selected online plasticity learning rate: 0.70.

The routing advantage also replicated in the earlier two-seed search. Seed 41 routed 16×32 achieved 100% / 87.5% / 70.625% on 4 / 8 / 16 facts; seed 42 achieved 90% / 82.5% / 66.25%.

The preference phase used synthetic/programmatic labels. It must not be described as human-feedback RLHF.

## Next validation required

Real public text/code pretraining, established continual-learning and long-context benchmarks, stronger memory baselines, more seeds/ablations, full structured-value recall, and human preference labels if literal RLHF is required.
