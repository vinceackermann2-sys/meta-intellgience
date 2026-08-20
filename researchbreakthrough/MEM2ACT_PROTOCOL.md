# Mem2Act protocol freeze — 2026-08-20

## Development / final-test split

The public release contains 400 QA records in release order.

- QA 001–100 are the **development set**. Gold arguments from these records have already been used for diagnostics and architecture iteration, so they must never be presented as an untouched final result. The released conversation file resolves 94/100 of these tasks.
- QA 101–400 are the **held-out final set**. Before this protocol freeze, their gold tool arguments, grounding labels, and answers have not been read by the research scripts used for architecture design. Only schema structure, current query text, and provenance/session-mapping metadata have been inspected. Keep their gold arguments sealed until the design is frozen.

The final set contains 300 QAs, of which 287 are expected to be resolvable from the released conversation file because the full-release provenance audit found 381/400 resolvable and the dev set accounts for 94/100.

## Public-release mapping defect

A provenance-only audit of all 400 QAs found:

- 381/400 source mappings resolve to `Mem2ActBench/toolmem_conversation.jsonl`;
- 19/400 do not resolve;
- the missing source IDs are absent from both `original_conversation_ids` and per-turn `source_id` indexes;
- all 19 final QA texts can be matched to their pre-normalized records in `processed_data/memory_driven_qa_Kimi-K2-Instruct-0905_v2.jsonl`;
- the authors' `05_benchmark_normalization.py` copies each QA's `source_records` into `source_conversation_ids`, while the session file is constructed independently from `conversation_sequence.csv`. This permits a QA source record to survive even when its conversation is absent from the released session sequence.

Therefore report two distinct evaluation surfaces:

1. **Strict released-input score:** only the 381 QAs whose required source provenance exists in the released session file; never silently drop the other 19 when describing the full benchmark.
2. **Repaired-release score:** only after reconstructing missing input histories from upstream public records, with the reconstruction procedure published. This score must be clearly labeled and must not be conflated with the authors' original internal evaluation.

## Development baselines on the same 94 resolvable QA 001–100 tasks

Backbone: `Qwen/Qwen2.5-0.5B-Instruct`.

- Flat parameter-conditioned semantic top-6 retrieval: macro task F1 = **0.97**.
- Typed action-event slot compiler + parameter-conditioned episodic evidence + schema state: macro task F1 = **28.69**, exact full argument set = 20.21%, correct/gold parameter rate = 25.12%.

This ~29.7x relative same-model improvement is strong causal evidence that representation matters, but it is a development result, not a benchmark claim.

## Current candidate under test

Pointer-grounded slot execution:

- compile exact candidates from prior typed tool/action events, schema defaults/enums, current-request spans, and semantically relevant episodic spans;
- model selects a candidate ID when possible rather than regenerating the value;
- unresolved values use an explicit `derive` path;
- deterministic executor copies candidate values losslessly, projects only valid schema keys, and coerces schema types;
- no gold argument, `grounding_info`, or `evolution_chain` may construct candidates or affect selection.

## Metric alignment

The ACL paper describes F1 as parameter-level precision/recall and gives the ground-truth tool in its main parameter-grounding experiment. The next accepted run must report:

- global parameter precision;
- global parameter recall;
- global parameter F1;
- per-task macro F1 (diagnostic only);
- exact argument-set accuracy;
- slot accuracy by explicit/inferred/default and value complexity when labels are opened for final scoring.

Do not open QA 101–400 gold arguments until the candidate architecture, prompts, deterministic transforms, and retrieval budgets are frozen.