# Memory-to-Action Breakthrough Gate v2

Status: pre-registered on 2026-08-20 BEFORE reading any QA101-400 gold labels.

## Why v2 exists
The original QA071-100 validation slice was inspected repeatedly during Gate-A debugging. It is therefore treated as development from this point forward. We do not claim future architecture choices are validated by QA071-100.

No QA101-400 `tool_call.arguments`, `grounding_info`, or `evolution_chain` labels have been opened for the v2 split at preregistration time.

## Fixed split
- Development: QA001-100. Architecture, training, candidate generation, hyperparameters, and diagnostics may use this split.
- Gate-A validation: exactly 50 QA IDs sampled without replacement from QA101-400 using Python `random.Random(20260820).sample(range(101,401), 50)`, then sorted.
- Final sealed Gate-B test: the remaining 250 QA IDs from QA101-400.

### Gate-A validation IDs
106, 114, 123, 124, 125, 128, 136, 137, 138, 143, 161, 178, 196, 200, 208, 211, 228, 239, 247, 249, 250, 251, 259, 260, 262, 264, 269, 271, 299, 300, 307, 310, 313, 320, 321, 324, 336, 346, 356, 365, 368, 369, 371, 380, 384, 392, 396, 397, 399, 400.

The final Gate-B test is the complement of those IDs within 101..400.

## Scientific claim under test
A memory-to-action system that preserves episode provenance, resolves typed memory addresses/operations before dereferencing values, and executes exact/default/normalization operations deterministically can outperform conventional generative long-term-memory pipelines at parameter grounding while using substantially less generative inference.

## Gate A — freeze candidate on untouched 50-QA validation
All must hold on the fixed Gate-A validation IDs using the same parameter-level precision/recall F1 evaluator:
1. Global parameter F1 >= 0.60.
2. Explicit slot exact accuracy >= 0.50.
3. Inferred slot exact accuracy >= 0.25.
4. Default slot exact accuracy >= 0.85.
5. No inference feature may contain gold answer values, gold provenance, `grounding_info`, `evolution_chain`, or QA source IDs used as retrieval labels.
6. Gold values/presence may be TRAINING LABELS on QA001-100 but never input features.
7. Exact identifiers/URLs/codes must be copied/dereferenced from memory when available rather than regenerated.
8. Record candidate code, model, hyperparameters and commit SHA before scoring the 50-QA validation.

If Gate A fails, the 250-QA final set remains sealed. Any architecture change after observing Gate-A validation makes those 50 development data; a new preregistered validation must then be carved from the still-sealed final pool before another claim-quality Gate A.

## Gate B — one-shot sealed 250-QA evaluation
Run the frozen candidate exactly once on the remaining 250 IDs.

A breakthrough candidate requires:
1. Parameter F1 > 0.3593 published A-Mem + Qwen2.5-72B reference, with protocol/subset compatibility stated explicitly.
2. Prefer F1 >= 0.40 practical margin.
3. Report explicit/inferred/default accuracy, exact argument-set rate, latency, memory size, embedding calls and generative calls/tokens.
4. If any generative residual is used, report its call fraction and a no-generative ablation.
5. No post-hoc tuning on the 250 final IDs.

Because Gate B uses a preregistered 250-QA subset rather than the published full 400, direct score comparison is indicative, not perfectly apples-to-apples. A later frozen full-400 run may be reported separately but cannot be called a sealed result.

## Gate C — independent reproduction
A Mem2Act result alone is not a confirmed breakthrough. Reproduce the core mechanism on an independent task family such as MemoryAgentBench TTL/Conflict, ForgetEval-Adv, or another action-grounding benchmark. Require strong baselines, no benchmark-specific answer heuristics, migration/deletion/tenant-isolation tests when relevant, and efficiency reporting.

## Claim language
- Before Gate A: research direction / hypothesis.
- Gate A passed: frozen candidate, not breakthrough.
- Gate B passed: breakthrough candidate.
- Gate C independently reproduced: defensible breakthrough claim, subject to external replication/peer review.
