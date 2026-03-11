# forge/identify — Teacher Model Identification

## Purpose

The `identify` stage is the entry point of the UOR-FORGE pipeline. Its responsibility is to enumerate, profile, and formally register the frontier teacher models that will contribute reasoning vectors to AMALGEMEN. No extraction, merging, or distillation occurs here. This stage exists solely to establish a verified, structured record of what teachers are available, what reasoning domains they cover, and whether they are compatible with the forge's coordinate substrate.

Identification is not a trivial step. The semantic map that ships with AMALGEMEN traces every capability back to a specific teacher at a specific version. That provenance chain begins here. A teacher that is not formally registered in this stage cannot contribute to any downstream stage.

## What This Stage Does

The `identify` stage performs four operations in sequence:

| Operation | Description |
|---|---|
| **Enumeration** | List all candidate teacher models with their source, version, parameter count, and architecture family |
| **Domain profiling** | Map each teacher's known reasoning strengths to UOR coordinate regions in the cat9 topology |
| **Compatibility check** | Verify that the teacher's architecture and tokenisation are compatible with the forge's extraction protocol |
| **Registration** | Write a signed teacher manifest entry that downstream stages can reference by ID |

## Inputs

| Input | Format | Source |
|---|---|---|
| Candidate teacher model list | YAML or JSON manifest | Human-authored or externally sourced |
| cat9 topology coordinate map | Rust-generated from `foundation/` | `cargo run --bin uor-build` |
| Architecture compatibility matrix | JSON | Maintained in this directory |

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| Teacher manifest | `teachers.json` — one entry per registered teacher | `forge/extract/`, `forge/donors/` |
| Domain coverage map | `coverage.json` — UOR coordinate regions per teacher | `forge/merge/`, `forge/eval/` |
| Compatibility report | `compat.md` — human-readable pass/fail per teacher | Review and audit |

## Teacher Manifest Schema

Each entry in `teachers.json` records the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique identifier for this teacher (e.g., `teacher-alpha-v1`) |
| `source` | string | Model source or organisation |
| `version` | string | Exact model version or checkpoint hash |
| `parameters` | integer | Parameter count (target: 120B-class) |
| `architecture` | string | Architecture family (e.g., `llama`, `mistral`, `qwen`) |
| `domains` | array | UOR coordinate regions this teacher is profiled to cover |
| `compatible` | boolean | Whether the teacher passed the compatibility check |
| `registered_at` | ISO 8601 | Timestamp of registration |

## Invariants

A teacher may only proceed to `forge/extract/` if all of the following hold:

1. `compatible` is `true` in the teacher manifest.
2. At least one UOR coordinate domain is assigned in `coverage.json`.
3. The teacher entry is signed and present in `teachers.json`.

## Relationship to Other Stages

```
forge/identify/
    │
    ├── teachers.json ──────────────────► forge/extract/
    ├── teachers.json ──────────────────► forge/donors/
    └── coverage.json ──────────────────► forge/merge/
                      ──────────────────► forge/eval/
```

---

*Stage 1 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
