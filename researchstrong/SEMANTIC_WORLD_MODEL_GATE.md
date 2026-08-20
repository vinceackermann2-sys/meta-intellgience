# Semantic World Model Pivot — preregistered development gates

Status: research only. No breakthrough claim. Do not merge.

## Why this pivot exists
The prior Mem2Act selector family plateaued around 49% QA001-100 development CV despite high episodic evidence availability. The unresolved failure is semantic binding across schemas (e.g. company -> ticker, country phrase -> country code, historical field `collection` -> future field `id`).

Recent 2026 prior art already covers personal world models, entity/property/time memory, typed memory, provenance graphs, executable memory and trace compilation. Therefore **semantic world model** is not a novelty claim by itself. The narrower hypothesis is that a model-independent longitudinal world state can bridge unrelated historical schemas to future tool schemas while exact remembered values bypass generation.

## Locked data protocol
- QA001-100: development only.
- QA101-400 gold: remains unopened on this branch until all development gates below pass.
- Parent `BREAKTHROUGH_GATE_V2.md` remains binding for the later hidden validation/final split.
- Candidate values may be executor payloads and development labels, but must never be semantic-role/ranking features.
- `grounding_info` is scoring/reporting only.

## Gate SWM-A — world-state formation
On QA001-100 development oracle analysis:
- explicit exact-value/property availability >= 85%
- inferred exact-value/property availability >= 40%
- provenance must identify source episode/turn/field or a deterministic ontology transform

Rationale: the previous inferred candidate space had only 25% coverage, equal to the old threshold and therefore no safety margin.

## Gate SWM-B — cross-schema semantic binding
5-fold QA-level CV on QA001-100:
- explicit top-1 exact dereference >= 50%
- inferred top-1 exact dereference >= 25%
- candidate value itself masked from all semantic-role features
- improvement must exceed the strongest prior flat address/role resolver, not merely candidate oracle coverage

## Gate SWM-C — action compiler
Only after SWM-A and SWM-B pass, compose semantic world state with deterministic default/omit/normalization execution. Require on QA001-100 grouped CV:
- parameter F1 >= 60%
- explicit accuracy >= 50%
- inferred accuracy >= 25%
- default accuracy >= 85%

Only then may the preregistered hidden Gate-A validation subset from `BREAKTHROUGH_GATE_V2.md` be evaluated.

## Breakthrough claim remains stricter
Passing these development gates is **not** a breakthrough. A defensible breakthrough candidate still requires:
1. untouched Mem2Act validation/final superiority under a compatible metric,
2. efficiency/ablation evidence showing the gain comes from world-state compilation rather than more context/model compute,
3. independent reproduction on a second longitudinal personalized action benchmark (reserved target: UserToolBench; examples/answers must remain unopened during Mem2Act design),
4. reversible correction/deletion and tenant isolation,
5. recompilation/migration across materially different model representations with negligible loss,
6. novelty analysis against Personal World Models, MindMemOS, MEMORA, MemIR, MemCompiler, TraceCompiler and related 2026 work.
