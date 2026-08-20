# Breakthrough Gate

This file defines the pass/fail criteria *before* final benchmark results are accepted. The purpose is to prevent post-hoc goal shifting.

## Candidate contribution

A model-independent canonical event ledger is compiled online into multiple operation-specific executable memory states rather than a single flat retrieval store:

1. typed action/slot state for exact tool parameters and reusable action state;
2. versioned temporal/current-state registers for changing facts;
3. relation/program graphs for compositional queries;
4. semantic state for stable abstractions;
5. episodic residuals for evidence that cannot yet be safely compiled;
6. optional model-specific fast/parametric caches that can be rebuilt from the canonical ledger after a model upgrade.

The query/task determines which memory substrates are read and composed. The canonical ledger remains the source of truth, so correction, deletion, provenance, tenant isolation, and model migration remain possible.

This is not claimed to be novel merely because it is hierarchical or hybrid. MemOS, MemoryOS, EverMemOS, MemoTime, MemCog, UniMem, CoMem and related systems already cover important pieces of that design space. Any claim must be based on measured capability/efficiency that existing systems do not demonstrate.

## Gate A — real action grounding (flagship)

Benchmark: official Mem2ActBench protocol, all released evaluation tasks accounted for. No silent dropping of unresolved sessions.

Required baselines using the same backbone and decoding budget:
- no memory / current query + schema only;
- full available context when it fits;
- flat semantic top-k retrieval;
- BM25/dense/hybrid passive retrieval;
- typed compiler with each substrate ablated independently.

External reference points from the Mem2ActBench paper:
- Qwen2.5-7B + A-Mem: 30.99 F1;
- best reported passive retrieval ablation: 30.7 F1;
- oracle/perfect retrieval: 53.8 F1.

### Strong efficiency signal
A 0.5B backbone reaches >= 31.0 F1 under a comparable official protocol while significantly reducing prompt/memory inference compute versus the reported 7B systems.

### Breakthrough-level action signal
At least one of:
- Qwen2.5-0.5B + compiler reaches >= 35 F1 on the full official evaluation; or
- Qwen2.5-7B + compiler reaches >= 36 F1 (>= +5 absolute over 7B A-Mem); or
- the compiler matches/exceeds a stronger system while reducing end-to-end model FLOPs or latency by >= 5x and stored/retrieved token state by >= 5x.

Additionally:
- explicit, inferred, default and complex-identifier slot accuracy must all be reported;
- exact tool argument evaluation must be used, not only semantic similarity;
- tool-selection accuracy cannot regress materially;
- no QA label, grounding_info, evolution_chain label, or gold tool arguments may influence retrieval/compilation at test time.

## Gate B — experienced-agent memory

Benchmark: LongMemEval-V2 or another public trajectory-memory benchmark that measures environment state, workflow knowledge, dynamic state, gotchas and premise awareness.

Current public reference point: AgentRunbook-C reports 72.5% average accuracy; strongest RAG baseline reported in that work is 48.5%.

Breakthrough-level corroboration requires one of:
- > 72.5% under the benchmark's comparable evaluation protocol with lower latency/state cost; or
- within 2 absolute points of the best reported accuracy while improving measured retrieval/inference latency or stored/retrieved state by >= 5x.

The memory compiler must ingest raw trajectories/events, not benchmark-derived gold summaries.

## Gate C — continual change, correction and migration

The following are mandatory systems tests, not optional demos:

1. >= 10,000 sequential memories per tenant with a mixture of new facts, updates and contradictions.
2. >= 1,000 corrections/retractions/deletions with audited removal of the superseded compiled contribution.
3. Temporal queries distinguish historical truth from current belief.
4. Cross-tenant contamination = 0 in deterministic isolation tests and randomized stress tests.
5. Model-version migration rebuilds model-specific state from the canonical ledger, not by copying incompatible neural tensors.
6. After migration to a materially different representation/model, retained-memory accuracy drops < 1 absolute point relative to the pre-migration system on the same canonical records, or the loss is explicitly explained by the new model's base capability.
7. Writes arriving during migration are replayed from a sequence-numbered log and no acknowledged write is lost.

## Gate D — causal ablation

A result does not count as an architectural breakthrough unless the gain can be causally localized.

Required ablations:
- remove typed action state;
- remove temporal/versioned registers;
- remove relation/program graph;
- remove episodic residual retrieval;
- replace operator-conditioned routing with a single flat top-k retriever;
- equalize total retrieved tokens;
- equalize backbone, decoding parameters and available source information.

At least one proposed new mechanism must produce a statistically reliable gain on the task it is designed for. If the full system wins but no individual mechanism survives equal-budget ablation, the result is engineering integration rather than a demonstrated new memory principle.

## Gate E — replication and statistical reliability

Before using the word breakthrough:

- evaluate the final frozen design on at least two public benchmark families;
- run >= 3 independent seeds where stochastic training/routing is involved;
- report confidence intervals or bootstrap intervals on benchmark-level deltas;
- reproduce the flagship result on a second model family or materially different model size;
- preserve a genuinely untouched final test split or benchmark until design decisions are frozen;
- publish exact scripts, revisions, prompts, model IDs, decoding parameters and failure counts.

## Claim ladder

- **Interesting experiment:** mechanism works on synthetic or internal tasks.
- **Promising result:** beats a strong baseline on one public benchmark under a clean protocol.
- **Research contribution:** survives strong baselines, ablations and replication on >= 2 benchmark families.
- **Breakthrough:** establishes a new capability/efficiency frontier that current systems do not demonstrate, while also passing Gates C, D and E.

## Current status

- Synthetic RSM continual-memory dynamics: promising internal evidence only.
- Tenant isolation, reversible updates and model migration: demonstrated at small synthetic scale; Gate C not yet passed at 10k+ real memories.
- MQuAKE-Remastered 90.42% per-case structured-triple transfer: useful language/program-transfer result, **not** comparable to the true simultaneous multi-edit benchmark and not end-to-end because it consumes `new_triples_labeled`.
- Banking77 compact-state work: partial rejection; compact cortex+exceptions did not beat full episodic k-NN.
- Mem2Act offline location analysis: supports typed + semantic + schema hybrid memory; this is diagnostic, not benchmark performance.
- Mem2Act 0.5B flat-retrieval and typed-compiler A/B runs: in progress at time this gate was written.

Do not weaken these gates after seeing results. If a gate fails, change the architecture or narrow the claim.