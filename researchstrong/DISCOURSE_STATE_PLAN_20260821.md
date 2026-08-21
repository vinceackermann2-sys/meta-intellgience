# Discourse Referent State Diagnostic

Development-only research plan. Do not merge.

The semantic world-model ingest gate passed (93.41% explicit / 42.50% inferred candidate availability), while the anchor-scoped role binder failed SWM-B (34.53% CV F1; 44.40% explicit / 14.20% inferred). This isolates entity/topic reference resolution and role binding as current bottlenecks.

Next diagnostic, before any QA101-400 gold is read:
1. Compile query-independent discourse/entity clusters from world nodes using episode membership and bounded shared-identity links.
2. Maintain query-independent salience features: recency, recurrence, structured support, and generic user-language salience markers.
3. At query time, select referent clusters using current intent + target semantic type/role plus the salience state. Candidate values are never ranking features.
4. Measure gold-property reachability in top-1/2/3/5 discourse clusters, and union with exact entity anchors.
5. Continue only if this materially improves over the current exact-anchor/shared-value reachability (71.43% explicit / 37.50% inferred). Otherwise reject discourse salience as the missing primitive.

Novelty guardrail: contextual-intent/coreference memory already exists in 2026 work. This diagnostic is a control mechanism, not a novelty claim. The research contribution, if any, must come from exact longitudinal world-state-to-action compilation with reversible/model-portable memory and external benchmark gains.

QA001-100 only for development. QA101-400 gold remains sealed under BREAKTHROUGH_GATE_V2.md.
