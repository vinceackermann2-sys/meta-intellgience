# Zero-LLM Mem2Act Slot VM — Development Result

**Status:** development-only diagnostic. This is **not** an official Mem2ActBench test result and is **not** a breakthrough claim.

## Protocol

- Benchmark surface: QA001–100 in release order.
- Public-release sessions resolvable: 94/100.
- QA101–400 gold tool arguments remain sealed and were not inspected.
- Correct target tool/schema is supplied, matching the paper's parameter-grounding isolation setup.
- No embedding model and no LLM are used to predict arguments.
- Gold arguments are read only after prediction for scoring.

## Executor

For each schema slot, the deterministic VM uses this precedence:

1. latest historical tool-call value with the same slot name;
2. machine-readable JSON schema default;
3. default parsed from schema description;
4. general schema/type policy learned on QA001–100 only.

The general policy contains no task-specific answer constants:

- optional array/list -> `[]`;
- optional string -> `""`;
- boolean -> `false`;
- offset -> `0`;
- page/pageIndex -> `1`;
- an index explicitly described as latest / starting at zero -> `0`.

## Result

| Metric | Result |
|---|---:|
| Resolvable tasks | 94 |
| Correct parameters | 95 |
| Predicted parameters | 215 |
| Gold parameters | 207 |
| Global precision | 44.186% |
| Global recall | 45.894% |
| **Global parameter F1** | **45.024%** |
| Exact argument-set accuracy | 13.830% |

Grounding-type parameter accuracy:

| Grounding type | Accuracy |
|---|---:|
| Default | 79.268% |
| Explicit | 28.409% |
| Inferred | 13.514% |

Prediction sources:

- general policy: 101 slots;
- latest same-name historical slot: 49;
- JSON schema default: 35;
- parsed description default: 30.

## Interpretation

This is strong evidence that a large portion of Mem2Act parameter grounding is a deterministic state/schema execution problem rather than a free-form language-generation problem. It does **not** establish superiority to published systems because this is a tuned development surface with public-release session gaps and the evaluator/protocol still needs exact reproduction against the benchmark authors' implementation.

The same deterministic rule that works well for defaults performs poorly for semantically grounded state: explicit-slot accuracy is only 28.4%, and inferred-slot accuracy is 13.5%. Naive latest-value-by-slot therefore cannot be the final architecture.

The next candidate is a selective hybrid Memory Action IR:

- `DEFAULT(rule)` — deterministic, no model call;
- `COPY(pointer)` — exact byte-preserving value from provenance-aware memory;
- `NORMALIZE(pointer, schema_operator)` — deterministic or small-model canonicalization;
- `INFER(evidence)` — model invoked only for genuine semantic derivation;
- `OMIT` — schema-aware omission.

The intended design executes high-confidence defaults/copies without an LLM and routes only ambiguous explicit/inferred residuals through semantic selection or a small model. This should be evaluated both for parameter F1 and for actual model calls, generated tokens, prompt tokens, latency and FLOPs.

## Related dev oracle

On the same 94 resolvable development tasks (207 gold parameters), the exact-value candidate ceiling is:

- base pointer/schema candidate oracle: 67.633%;
- after general schema policies: **79.710%**;
- default slots: 60.976% -> **91.463%**;
- explicit slots: 87.500%;
- inferred slots: 35.135%.

This suggests the hard residual is no longer basic retention/default execution; it is semantic candidate selection and schema-conditioned derivation.
