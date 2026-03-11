# forge/extract — Reasoning Vector Extraction

## Purpose

The `extract` stage draws high-signal reasoning vectors from each registered teacher model and assigns each vector a coordinate address in the cat9 topology. This is the first stage at which the UOR semantic layer becomes active. Raw weight tensors enter; addressed reasoning vectors leave.

Extraction is the most computationally intensive stage in the pipeline. It operates directly on teacher model weights and must produce outputs that are fully addressable — every vector that exits this stage carries a valid UOR coordinate, or it is discarded. Unaddressed vectors are not passed downstream under any circumstances (see FORGE.md, Invariant 2).

## What This Stage Does

The `extract` stage performs three operations per teacher:

| Operation | Description |
|---|---|
| **Signal identification** | Locate high-signal weight regions within the teacher using activation analysis, gradient attribution, or equivalent methods |
| **Vector extraction** | Isolate the identified regions as discrete reasoning vectors |
| **Coordinate assignment** | Run each vector through the UOR resolution pipeline to assign a cat9 coordinate address |

Coordinate assignment uses the resolution pipeline defined in the cat9 topology: the vector is typed, resolved under the dihedral group D_{2^n}, classified into the four-component partition (irreducible, reducible, unit, exterior), and certified. Only vectors that resolve to a valid coordinate are retained.

## Inputs

| Input | Format | Source |
|---|---|---|
| Teacher manifest | `teachers.json` | `forge/identify/` |
| Teacher model weights | Model-native format (safetensors, bin, etc.) | External — not stored in this repo |
| cat9 topology | Rust traits from `foundation/` | `cargo run --bin uor-build` |
| Extraction config | `extract.toml` — per-teacher extraction parameters | Maintained in this directory |

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| Addressed vector set | `vectors/{teacher-id}/` — one file per extracted vector | `forge/merge/` |
| Extraction report | `report/{teacher-id}.md` — yield, discard rate, coordinate distribution | `forge/eval/` |
| Resolution certificates | `certs/{teacher-id}/` — one UOR certificate per vector | `forge/merge/`, `forge/eval/` |

## Vector Record Schema

Each extracted vector is stored with the following metadata:

| Field | Type | Description |
|---|---|---|
| `vector_id` | string | Stable unique identifier (hash of content + coordinate) |
| `teacher_id` | string | Reference to the source teacher in `teachers.json` |
| `coordinate` | string | UOR coordinate address in the cat9 topology |
| `partition_class` | enum | `irreducible` \| `reducible` \| `unit` \| `exterior` |
| `cert_id` | string | Reference to the resolution certificate in `certs/` |
| `signal_score` | float | Normalised signal strength score (0.0–1.0) |
| `extracted_at` | ISO 8601 | Timestamp of extraction |

## Extraction Config

`extract.toml` controls the extraction behaviour per teacher:

```toml
[teacher-alpha-v1]
method = "activation_attribution"   # or "gradient_saliency", "probe_based"
layers = "all"                       # or a range, e.g. "16-32"
signal_threshold = 0.75              # minimum signal score to retain a vector
max_vectors = 50000                  # cap on vectors extracted per teacher
```

## Invariants

1. Every vector in `vectors/` must have a corresponding certificate in `certs/`. Vectors without certificates are deleted at the end of the stage.
2. Every vector must have a `partition_class` assigned. Vectors that fail resolution are logged in the extraction report and discarded.
3. The extraction report must record the total number of candidates evaluated, vectors retained, and vectors discarded, with discard reasons.

## Relationship to Other Stages

```
forge/identify/
    │
    └── teachers.json ──────────────────► forge/extract/
                                              │
                                              ├── vectors/ ──────────────► forge/merge/
                                              ├── certs/   ──────────────► forge/merge/
                                              └── report/  ──────────────► forge/eval/
```

---

*Stage 2 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
