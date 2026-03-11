# forge/merge — Semantic Vector Fusion

## Purpose

The `merge` stage is where the core intellectual work of UOR-FORGE occurs. It takes the addressed reasoning vectors produced by `forge/extract/` — from multiple teachers, across multiple coordinate regions — and fuses them into a unified, non-redundant vector set that will be used to train AMALGEMEN.

This is not standard model merging. No weight averaging, SLERP interpolation, or magnitude-based heuristics are applied here. All fusion decisions are expressed as HARMONIA-DSL programs and operate on semantic coordinates, not tensor indices. The result is a fused vector set where every entry is traceable to its source teacher(s), its coordinate address, and the certificate that attests the fusion was valid (see FORGE.md, Invariants 3 and 4).

## What This Stage Does

The `merge` stage performs four operations:

| Operation | Description |
|---|---|
| **Proximity scoring** | Compute semantic distance between vectors from different teachers in UOR coordinate space using HARMONIA-DSL proximity operators |
| **Conflict detection** | Identify coordinate regions where multiple teachers provide vectors with low proximity (semantic disagreement) |
| **Fusion** | Combine vectors guided by proximity scores and confidence weights using HARMONIA-DSL fusion operators |
| **Certification** | Issue a UOR certificate for every fusion decision, linking the fused vector to its sources and the operators applied |

Fusion decisions are not heuristic. Each is a HARMONIA-DSL program that can be replayed, audited, and independently verified against the cat9 topology.

## Inputs

| Input | Format | Source |
|---|---|---|
| Addressed vector sets | `vectors/{teacher-id}/` | `forge/extract/` |
| Resolution certificates | `certs/{teacher-id}/` | `forge/extract/` |
| Domain coverage map | `coverage.json` | `forge/identify/` |
| HARMONIA-DSL operator library | `harmonia/` submodule | `harmonia/` |
| Merge config | `merge.toml` — fusion strategy parameters | Maintained in this directory |

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| Fused vector set | `fused/` — one file per fused vector | `forge/distill/` |
| Fusion certificates | `fusion-certs/` — one UOR certificate per fusion decision | `forge/distill/`, `forge/eval/` |
| Provenance index | `provenance.json` — maps every fused vector to its source teachers and coordinates | `forge/amalgemen/`, `forge/eval/` |
| Conflict report | `conflicts.md` — coordinate regions with unresolved semantic disagreement | Review and audit |

## Fused Vector Record Schema

| Field | Type | Description |
|---|---|---|
| `fused_id` | string | Stable unique identifier for the fused vector |
| `coordinate` | string | UOR coordinate address (inherited or resolved from sources) |
| `source_vectors` | array | List of `vector_id` entries from `forge/extract/` |
| `source_teachers` | array | List of `teacher_id` entries contributing to this fusion |
| `fusion_cert_id` | string | Reference to the fusion certificate in `fusion-certs/` |
| `proximity_score` | float | Semantic proximity of source vectors at time of fusion (0.0–1.0) |
| `fusion_operator` | string | HARMONIA-DSL operator used (e.g., `semantic-weighted-mean`, `dominant-source`) |
| `fused_at` | ISO 8601 | Timestamp of fusion |

## Merge Config

`merge.toml` controls the fusion strategy:

```toml
[strategy]
default_operator = "semantic-weighted-mean"
conflict_threshold = 0.40   # proximity below this triggers conflict logging
min_proximity = 0.25        # below this, vectors are not fused — dominant source wins
confidence_weighting = true # weight by signal_score from extract stage

[coverage]
require_full_coverage = true  # fail if any cat9 coordinate region has no fused vector
```

## Conflict Resolution Policy

When two or more teachers provide vectors for the same coordinate region with proximity below `conflict_threshold`, the merge stage does not silently average them. Instead:

1. The conflict is logged in `conflicts.md` with the coordinate, the teachers involved, and their proximity score.
2. The dominant-source operator selects the vector with the highest `signal_score` from `forge/extract/`.
3. A conflict certificate is issued noting the resolution method.

This policy ensures that AMALGEMEN never silently inherits contradictory reasoning from different teachers.

## Invariants

1. Every fused vector must have a fusion certificate. Fused vectors without certificates are deleted before the stage closes.
2. All fusion decisions must be expressed as HARMONIA-DSL programs. Direct weight-space operations are prohibited.
3. The provenance index must account for every fused vector. Gaps in provenance are a stage failure.
4. The conflict report must be non-empty if any conflicts were detected, and empty if none were.

## Relationship to Other Stages

```
forge/extract/
    │
    ├── vectors/ ───────────────────────► forge/merge/
    └── certs/   ───────────────────────► forge/merge/
                                              │
                                              ├── fused/          ──────► forge/distill/
                                              ├── fusion-certs/   ──────► forge/distill/
                                              │                   ──────► forge/eval/
                                              └── provenance.json ──────► forge/amalgemen/
                                                                  ──────► forge/eval/
```

---

*Stage 3 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
