# forge/donors — Teacher Model Donor Registry

## Purpose

The `donors` directory is the human-readable, version-controlled registry of all teacher models that have contributed — or are candidates to contribute — to AMALGEMEN. It is the longitudinal record of the forge's teacher relationships: who donated, what they donated, when, and what became of their contribution.

Unlike the machine-readable manifests produced by `forge/identify/`, the `donors` registry is designed for human audit, project governance, and historical reference. It answers questions that the pipeline artefacts do not: Why was this teacher selected? What was its known reasoning profile at the time of ingestion? Were there any concerns about its outputs? What fraction of AMALGEMEN's final capabilities originated from this teacher?

The `donors` registry is not a pipeline stage in the operational sense — it does not transform data or produce artefacts consumed by other stages. It is a **governance layer** that sits alongside the pipeline and provides the human context that certificates and manifests alone cannot.

## Directory Structure

Each registered donor has its own subdirectory:

```
forge/donors/
├── README.md                  # This file
├── registry.json              # Machine-readable index of all donors
└── {donor-id}/
    ├── profile.md             # Human-readable donor profile
    ├── manifest.json          # Machine-readable donor record (mirrors teachers.json entry)
    └── contribution.md        # Post-distillation record of what this donor contributed
```

## Donor Profile Schema

Each `profile.md` documents the following:

| Section | Content |
|---|---|
| **Identity** | Model name, version, source organisation, parameter count, architecture |
| **Reasoning profile** | Known strengths and weaknesses; domains where this teacher is considered frontier-class |
| **UOR coordinate coverage** | Which regions of the cat9 topology this teacher is expected to populate |
| **Selection rationale** | Why this teacher was chosen; what gap in the coordinate space it fills |
| **Known limitations** | Any documented failure modes, biases, or output quality concerns |
| **Ingestion history** | Date of registration, date of extraction, checkpoint version used |

## Registry Index

`registry.json` is a machine-readable index of all donors, structured as an array of objects with the following fields:

| Field | Type | Description |
|---|---|---|
| `donor_id` | string | Stable unique identifier (matches `teachers.json` `id` field) |
| `status` | enum | `candidate` \| `registered` \| `extracted` \| `contributed` \| `retired` |
| `profile_path` | string | Relative path to `profile.md` |
| `manifest_path` | string | Relative path to `manifest.json` |
| `contribution_path` | string | Relative path to `contribution.md` (null until distillation completes) |
| `registered_at` | ISO 8601 | Date of registration |
| `last_updated` | ISO 8601 | Date of last status change |

## Donor Lifecycle

A donor moves through the following states over the course of the project:

```
candidate ──► registered ──► extracted ──► contributed ──► retired
                                │
                                └── (failed compat check) ──► retired
```

| State | Meaning |
|---|---|
| `candidate` | Under consideration; not yet formally registered |
| `registered` | Passed compatibility check; present in `teachers.json` |
| `extracted` | Reasoning vectors have been extracted by `forge/extract/` |
| `contributed` | Vectors have been fused into AMALGEMEN and provenance is confirmed |
| `retired` | No longer active; either superseded, failed evaluation, or withdrawn |

## Contribution Record

Once distillation and evaluation are complete, each donor's `contribution.md` is populated with:

- The number of reasoning vectors extracted from this donor.
- The number of those vectors that survived fusion (i.e., appear in the fused vector set).
- The UOR coordinate regions where this donor's vectors are dominant in AMALGEMEN.
- The UOR coordinate regions where this donor's vectors were overridden by another teacher.
- A summary of the donor's net contribution to AMALGEMEN's capability profile.

This record is the human-readable counterpart to the machine-readable provenance index in `forge/merge/provenance.json`.

## Governance Notes

The `donors` registry is the appropriate place to record any concerns, disputes, or decisions about teacher model selection that do not fit naturally into pipeline artefacts. If a teacher is retired mid-pipeline, the reason must be documented in its `profile.md`. If a teacher's contribution is disputed during evaluation, the dispute and its resolution must be recorded in `contribution.md`.

## Relationship to Other Stages

```
forge/identify/
    │
    └── teachers.json ──────────────────► forge/donors/
                                              │
                                              └── registry.json ──────────► (human audit)
                                              └── {donor-id}/contribution.md ◄── forge/eval/
```

---

*Governance layer of the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
