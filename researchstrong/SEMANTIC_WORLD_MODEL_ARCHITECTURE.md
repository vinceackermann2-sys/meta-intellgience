# Semantic World Model Memory Architecture

Status: research specification. No breakthrough claim. Do not merge into production.

## 1. Core thesis

Long-term agent memory should not be a bag of retrieved text and should not require an LLM to regenerate exact remembered state.

The system maintains a **model-independent canonical experience ledger** and continuously compiles it into a **semantic world state**: persistent entities, concepts, properties, relations, temporal versions, procedures, and provenance. A future tool schema binds to semantic roles in that world state. Only after the role/address is selected does the executor dereference the exact stored value and apply a deterministic transform.

In short:

`experience -> canonical ledger -> semantic world state -> semantic role binding -> exact dereference -> deterministic action`

The LLM is a residual reasoner, not the database.

## 2. Why this exists

Earlier Mem2Act experiments established:

- Flat retrieval + Qwen2.5-0.5B: ~0.97 macro F1.
- Typed memory presentation + same 0.5B: ~28.69 macro F1.
- Deterministic typed execution: ~45% development F1.
- Strongest flat/pairwise address selector plateaued around ~49% development CV.
- Top-2 episode retrieval often contains the correct explicit value (>90% availability in the earlier diagnostic), but selecting the correct semantic role remains difficult.
- The old inferred candidate space had only 25% coverage.
- Qwen2.5-0.5B and 1.5B residual generation recovered 0/30 unresolved inferred development slots in the controlled residual test.

The bottleneck is therefore not simply storage or retrieval. It is **semantic binding across different representations and schemas**.

Examples:

- historical `company = Tesla` -> future `symbol = TSLA`
- historical phrase `expanding into Germany` -> future `country_code = DE`
- historical `collection = 0x...` -> future `id = 0x...`
- historical `artist = Konshens` -> future generic `id = Konshens`

These require understanding what a field/value *means*, not just matching strings.

## 3. Layer A — canonical experience ledger

The ledger is the source of truth. It is append-only except for explicit tombstone/correction events.

Each event records:

- tenant/user ID
- timestamp / logical version
- source application/tool
- source conversation/episode
- user intent
- assistant action
- tool call arguments
- tool outputs
- raw text/evidence
- confidence/source trust where available

Properties:

- exact provenance
- correction/deletion support
- tenant isolation
- replayability
- migration across model generations

No learned representation is authoritative. Every compiled state can be rebuilt from this ledger.

## 4. Layer B — semantic world-state compiler

The compiler runs independently of the future query/tool.

It converts experience into addressable world-state objects:

### 4.1 Entities
Persistent things such as:

- people
- organizations
- companies
- projects
- accounts
- places
- files
- products
- services
- APIs/tools

Entity nodes may have aliases but retain provenance for every alias.

### 4.2 Properties
Exact values attached to semantic roles, for example:

- `company.name = Tesla`
- `company.market_symbol = TSLA`
- `place.country = Germany`
- `place.country_code = DE`
- `project.assignee = Alex`
- `account.network = mainnet`

A property is not merely a value. It is:

`(entity, semantic_role, exact_value, time/version, provenance)`

### 4.3 Concepts
Concepts abstract recurring or semantically important content that may never appear as a structured tool field:

- atmosphere
- technology
- station
- Bitcoin
- quarterly report
- health benefits of boxing

Concept extraction must be query-independent. This prevents benchmark-specific candidate construction.

### 4.4 Relations
Typed links between world-state objects:

- `company -> trades_as -> symbol`
- `place -> has_country_code -> code`
- `person -> assigned_to -> project`
- `tool_output -> describes -> entity`
- `event -> produced -> property`

Relations should be evidence-backed and versioned rather than globally hallucinated.

### 4.5 Procedures / skills
Repeated successful action traces can compile into procedural state, but this is distinct from semantic facts. A procedure references semantic roles rather than hard-coding tenant-specific values where possible.

## 5. Layer C — semantic role space

External tools use arbitrary schemas. The world model should not permanently adopt every API field name as its ontology.

Instead, source fields and target fields map into canonical semantic roles.

Examples:

- `ticker`, `symbol`, `stock_symbol` -> `market_identifier`
- `country_code`, `region_code` -> context-specific geographic-code roles
- `collection`, `contract`, `asset_id` -> possible entity identifiers depending on provenance/context

A source field is represented by a value-masked semantic address containing:

- originating user intent
- source tool
- field path
- sibling field roles
- entity/record type
- episode/source ID
- temporal position

A target tool field is represented by:

- current user intent
- target tool semantics
- field name
- description
- type/enum/default metadata

The binder learns or infers compatibility between these *roles*, not between answer strings.

## 6. Layer D — binder / query compiler

Given a current request and target tool schema:

1. infer the requested semantic operation/role for each necessary argument;
2. identify the relevant entity/world-state scope;
3. rank compatible semantic properties/relations;
4. choose `OMIT`, `DEFAULT`, `DEREFERENCE`, `NORMALIZE`, or `DERIVE`;
5. return a symbolic execution plan, not a generated argument value.

Example:

Current request: `Get Tesla's latest filing.`

Target schema: `symbol: string`

Possible plan:

`resolve_entity("Tesla") -> property(market_identifier) -> dereference_exact -> symbol`

The binder never needs to type `TSLA` itself.

## 7. Layer E — exact executor / Action VM

The executor owns structure and value fidelity.

Primitive operations:

- `OMIT(slot)`
- `DEFAULT(slot, schema_rule)`
- `COPY(slot, property_pointer)`
- `NORMALIZE(slot, property_pointer, deterministic_operator)`
- `DERIVE(slot, typed_relation_or_program)`
- `ASK/ESCALATE(slot)` when confidence is insufficient

Examples:

- `COPY(symbol, entity:Tesla.market_identifier)`
- `NORMALIZE(state, entity:California.name, US_STATE_TO_CODE)`
- `DEFAULT(offset, 0)`

Exact URLs, IDs, codes, account numbers, strings, and dates are dereferenced from canonical state. They are not regenerated token-by-token.

## 8. Layer F — residual reasoner

A language model is invoked only when the symbolic system cannot resolve an operation safely.

The residual reasoner should receive:

- a small scoped world-state packet
- explicit unresolved semantic role
- allowed output type/constraints
- provenance-backed evidence

It should not receive the whole history or a huge noisy candidate pool.

Repeated residual solutions can be proposed to the compiler as new semantic relations/procedures, but they remain provisional until validated by evidence or successful execution.

## 9. Online learning

New experience updates the architecture in two different ways:

### Exact state update
A new observation changes a versioned property.

Example:

`preferred_currency: USD -> SEK`

The current compiled property updates immediately; the old version remains in the canonical ledger/provenance history.

### Semantic learning
Repeated experiences can improve:

- entity resolution
- role equivalences
- relation schemas
- procedures
- confidence/calibration

Semantic learning is allowed to be approximate because exact values remain outside the learned representation.

## 10. Correction, deletion, and reversibility

Every compiled semantic object must retain dependencies on source ledger events.

If an event is corrected/deleted:

1. invalidate dependent properties/relations/concepts;
2. recompute affected compiled state;
3. preserve unaffected state;
4. issue a deletion/correction receipt if required.

This is why the canonical ledger remains authoritative rather than model weights.

## 11. Model migration

The world state is model-independent.

When the underlying LLM/encoder changes:

- keep canonical entities/properties/relations/provenance;
- recompute embeddings/indexes/learned role maps;
- recompile model-specific caches;
- do not attempt to copy opaque fast weights as durable memory.

This permits a user memory to survive a transition from model generation N to N+1.

## 12. Current empirical status

### SWM-A — formation gate
PASS on QA001-100 development oracle analysis:

- explicit exact-property availability: **93.41%**
- inferred exact-property availability: **42.50%**
- mean compiled properties/session: **769.71**
- mean semantic concept properties added/session: **238.56**

This proves only that the query-independent compiler can form enough potentially useful state. It does not prove the binder can select it.

### SWM-B — binding gate
Still pending at time of this specification.

Required 5-fold QA-level development CV:

- explicit top-1 exact dereference >= 50%
- inferred top-1 exact dereference >= 25%
- value-masked semantic features
- improvement over prior flat address/role resolvers

### SWM-C — action compiler gate
Only after SWM-B passes:

- parameter F1 >= 60%
- explicit >= 50%
- inferred >= 25%
- default >= 85%

Only after SWM-C may the preregistered untouched Mem2Act validation set be opened.

## 13. What would be scientifically interesting

This architecture is not novel merely because it is a world model, graph, typed memory, compiler, or pointer system; 2026 prior art already covers those ideas separately.

The potentially defensible contribution is the combination and empirical claim that **longitudinal personal experience can be compiled into model-independent semantic state that binds to unseen tool schemas and executes exact remembered values without generative reconstruction**, while supporting provenance, correction/deletion, tenant isolation, and cross-model migration.

That claim is not established until untouched validation and independent reproduction pass.
