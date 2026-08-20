# RSM Real-Data Benchmark Phase

## Current verdict
RSM is not yet a demonstrated scientific breakthrough.

The earlier synthetic result established strong exact keyed retention and low interference for routed fast weights. Real Banking77 paraphrase testing exposed a semantic-generalization failure.

## Real Banking77 4-intent pilot
40 real training utterances, 40 held-out real test utterances from PolyAI Banking77.

Using the scratch-trained 99,902,208-parameter RSM-100M slow model as representation:
- hard-routed RSM fast weights: 25.0% (chance)
- dense fast weights: 25.0% (chance)
- hidden centroid memory: 55.0%
- character n-gram 1-NN retrieval: 87.5%

Representation diagnostic using a 768-dimensional hashed character n-gram representation with the same online fast-weight update:
- dense fast weights: 85.0%
- hard-routed fast weights: 62.5%
- soft top-k routed fast weights: 77.5% mean across 3 router seeds
- 1-NN retrieval: 90.0%
- centroid: 82.5%

Interpretation: online plasticity can learn from real examples, but the scratch model representation and hard sparse routing are both bottlenecks. Soft routing helps but does not beat dense plasticity or retrieval.

## Scientific scale-up gates
Do not scale past 500M until all are met on a real pretrained base:
1. Semantic continual learning beats parameter-matched dense fast weights on Banking77/MemoryAgentBench TTL across >=3 seeds.
2. Conflict/update beats retrieval-only and dense fast-weight baselines on MemoryAgentBench Conflict Resolution and LongMemEval knowledge-update slices.
3. After 1k+ sequential updates, old-task accuracy loss is materially lower than LoRA/adapter fine-tuning at comparable update compute.
4. No material regression in held-out language-model perplexity or coding pass@1 after memory use.
5. Canonical-ledger recompilation preserves learned behavior across independently trained model versions.
6. Per-tenant isolation and deletion/supersession remain exact.
7. Report memory bytes/user, write/read latency, FLOPs, and context-token savings.

## Next benchmark ladder
A. Pretrained 160M pilot: Pythia-160M + dense fast weights + hard RSM + soft RSM + retrieval on 10-class Banking77.
B. Full MemoryAgentBench TTL and Conflict Resolution.
C. LongMemEval knowledge-update, temporal, multi-session, abstention.
D. TRACE continual-learning benchmark.
E. MBPP/HumanEval before/after sequential domain updates.
F. Only after reproduced wins: 500M, then 1B.

## Novelty bar
Fast weights/test-time memory alone are not novel. Titans/MIRAS and In-Place TTT already update neural memory/fast weights at inference. Any publishable claim must come from a mechanism or systems property that wins established benchmarks.
