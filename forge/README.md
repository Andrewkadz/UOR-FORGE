# forge/ — UOR-FORGE Distillation Pipeline

This directory contains the full UOR-FORGE distillation pipeline: the staged process by which reasoning vectors are extracted from frontier teacher models, semantically addressed using the cat9 topology, fused via HARMONIA-DSL operators, and assembled into AMALGEMEN — a 7B student model with a fully documented semantic map.

## Pipeline Overview

The pipeline is organised into seven stages, executed in sequence. Each stage has a dedicated subdirectory containing its configuration, intermediate artefacts, and a README that specifies its inputs, outputs, and invariants.

| Stage | Directory | Role |
|---|---|---|
| 1 | [`identify/`](identify/README.md) | Enumerate, profile, and register teacher models |
| 2 | [`extract/`](extract/README.md) | Extract high-signal reasoning vectors and assign UOR coordinates |
| 3 | [`merge/`](merge/README.md) | Fuse vectors from multiple teachers using HARMONIA-DSL operators |
| 4 | [`distill/`](distill/README.md) | Train AMALGEMEN 7B on the fused, addressed vector set |
| 5 | [`eval/`](eval/README.md) | Validate AMALGEMEN against capability benchmarks and semantic integrity checks |
| — | [`donors/`](donors/README.md) | Governance registry of all teacher model donors (not a pipeline stage) |
| 7 | [`amalgemen/`](amalgemen/README.md) | Assemble and release the final AMALGEMEN model and semantic map |

## Data Flow

```
forge/identify/
    │
    ├── teachers.json ──────────────────────────────────────────────────► forge/extract/
    ├── teachers.json ──────────────────────────────────────────────────► forge/donors/
    └── coverage.json ──────────────────────────────────────────────────► forge/merge/
                       ─────────────────────────────────────────────────► forge/eval/

forge/extract/
    │
    ├── vectors/ ───────────────────────────────────────────────────────► forge/merge/
    ├── certs/   ───────────────────────────────────────────────────────► forge/merge/
    └── report/  ───────────────────────────────────────────────────────► forge/eval/

forge/merge/
    │
    ├── fused/          ────────────────────────────────────────────────► forge/distill/
    ├── fusion-certs/   ────────────────────────────────────────────────► forge/distill/
    │                   ────────────────────────────────────────────────► forge/eval/
    └── provenance.json ────────────────────────────────────────────────► forge/amalgemen/
                        ────────────────────────────────────────────────► forge/eval/

forge/distill/
    │
    ├── checkpoints/      ──────────────────────────────────────────────► forge/eval/
    │                     ──────────────────────────────────────────────► forge/amalgemen/
    ├── alignment.md      ──────────────────────────────────────────────► forge/eval/
    │                     ──────────────────────────────────────────────► forge/amalgemen/
    └── distill-cert.json ──────────────────────────────────────────────► forge/amalgemen/

forge/eval/
    │
    ├── capability.md  ─────────────────────────────────────────────────► forge/amalgemen/
    ├── integrity.md   ─────────────────────────────────────────────────► forge/amalgemen/
    └── eval-cert.json ─────────────────────────────────────────────────► forge/amalgemen/

forge/amalgemen/
    │
    └── release/amalgemen-{version}/
            ├── weights/
            ├── semantic-map.json
            ├── semantic-map.md
            ├── certs/
            ├── manifest.json
            └── RELEASE.md
```

## Key Invariants

The following invariants apply across the entire pipeline. Each stage README specifies additional stage-local invariants.

1. **The cat9 topology is frozen.** No pipeline stage modifies `foundation/` or the 16 namespaces in `spec/src/namespaces/`.
2. **Every vector must be addressed.** No vector passes from `extract/` to `merge/` without a valid UOR coordinate.
3. **Every fusion must be certified.** No fused vector enters `distill/` without a UOR fusion certificate.
4. **HARMONIA-DSL is the only fusion interface.** All merge logic in `merge/` must be expressed as HARMONIA-DSL programs.
5. **Provenance must be complete.** The semantic map in `amalgemen/` must account for every capability. Gaps are release failures.
6. **The eval certificate gates release.** Nothing enters `amalgemen/release/` without a valid eval certificate from `eval/`.

## Coordinate Substrate

All pipeline stages operate against the UOR coordinate system defined in `foundation/` (cat9 topology, v6.3.0). The topology provides the address space within which all reasoning vectors are located, all fusion decisions are made, and all provenance is recorded.

## Operator Layer

All fusion operations in `merge/` are expressed using the HARMONIA-DSL operator library, available as a git submodule at `harmonia/`. No direct weight-space operations are permitted in the merge stage.

---

*See [FORGE.md](../FORGE.md) for the master project specification.*
