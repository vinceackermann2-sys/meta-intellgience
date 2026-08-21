# Memory-to-Action Breakthrough Gate v1

Status: pre-registered before opening QA101-400 gold.

## Scientific claim under test
A memory-to-action system that preserves episode provenance, resolves typed memory addresses before dereferencing values, and executes exact/default operations deterministically can outperform conventional generative long-term-memory pipelines at parameter grounding while using substantially less generative inference.

## Development split
- QA001-070: architecture/training only.
- QA071-100: validation and threshold selection only.
- QA101-400: sealed holdout. Do not inspect gold `tool_call.arguments`, `grounding_info`, or `evolution_chain` until the candidate is frozen.
- Context repair may use only released session data plus `qa_id` / `source_conversation_ids` to recover the exact public upstream conversation. QAs with no source IDs get empty memory.

## Gate A — freeze candidate on development data
All of the following must hold on QA071-100 using the same parameter-level precision/recall F1 definition used by our benchmark harness:
1. Global parameter F1 >= 0.60.
2. Explicit slot exact accuracy >= 0.50.
3. Inferred slot exact accuracy >= 0.25.
4. Default slot exact accuracy >= 0.85.
5. No feature may contain gold answer values, gold provenance, `grounding_info`, `evolution_chain`, or QA `source_conversation_ids` used as retrieval labels.
6. Exact identifiers/URLs/codes must be copied/dereferenced from memory rather than regenerated when present.
7. Record candidate code, hyperparameters, thresholds, model names, and commit SHA before the sealed run.

If Gate A fails, QA101-400 remains sealed and the architecture is redesigned.

## Gate B — one-shot sealed Mem2ActBench evaluation
Run the frozen candidate exactly once on QA101-400.

A breakthrough *candidate* requires:
1. Parameter F1 > 0.3593 (published A-Mem + Qwen2.5-72B result), subject to evaluator/protocol compatibility being verified.
2. Prefer F1 >= 0.40 as a practical margin, not a one-decimal tie.
3. Report explicit/inferred/default slot accuracy, exact argument-set rate, latency, memory size, embedding calls, and generative tokens/calls.
4. If a generative residual model is used, report its call fraction and compare against a no-generative ablation.
5. No post-hoc tuning on QA101-400 after observing its labels/results.

Published reference points from Mem2ActBench v1 paper: A-Mem Qwen2.5-72B F1 35.93; LTMemory 35.32; best passive hybrid retrieval 30.7; oracle retrieval 53.8.

## Gate C — reproduction beyond Mem2ActBench
A sealed Mem2ActBench win alone is not called a confirmed breakthrough. Reproduce the core mechanism on at least one independent memory task family (preferably more):
- MemoryAgentBench Test-Time Learning / Conflict Resolution,
- ForgetEval-Adv / selective forgetting,
- or another action-grounding benchmark with independently constructed data.

Required evidence:
1. Improvement over a strong retrieval/classifier/state baseline, not only a weak RAG baseline.
2. No benchmark-specific gold-derived heuristics.
3. Preserve model-independent canonical memory, exact correction/deletion, and tenant isolation where the task exercises them.
4. Report efficiency and failure cases.

## Claim language
- Before Gate A: research direction / hypothesis.
- Gate A passed: frozen candidate, not breakthrough.
- Gate B passed: breakthrough candidate.
- Gate C independently reproduced: defensible breakthrough claim, still subject to external replication/peer review.
