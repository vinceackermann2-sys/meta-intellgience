# 25M Architecture Research

This branch is an isolated compute experiment. It is not a claim of a breakthrough.

## Hypothesis

A parameter-matched autoregressive model augmented with surprise-gated, inference-time associative plasticity and a latent future-state prediction objective can improve long-horizon adaptation/memory without degrading held-out language/code loss relative to a same-scale Transformer.

## Primary comparison

- Baseline: 8-layer, d=512 causal Transformer.
- Candidate: same backbone plus fast associative memory updated online without backpropagation.
- Two seeds each.
- Same procedural pretraining/SFT/preference data budget and optimizer schedule.
- Target size: approximately 25M trainable parameters.

## Current pilot success gate

The candidate only advances if both seeds satisfy all of the following versus the Transformer seed-matched baseline:

1. Lower or equal final held-out loss (<= 1% regression allowed).
2. Higher mean long-horizon memory accuracy across 8/32/64/96-token distractor distances by >= 10 percentage points.
3. No numerical instability or failed checkpoint.

Passing this pilot is **not** itself a scientific breakthrough. It earns a second-stage run on real public text/code corpora and established continual-learning / long-context benchmarks. A breakthrough claim requires reproducible gains against stronger baselines and ablations.

## Post-training terminology

The script performs supervised fine-tuning followed by synthetic preference optimization (DPO-style). It must not be described as human-feedback RLHF because no human preference labels are used in this pilot.
