# forge/amalgemen — Final Model Assembly and Release

## Purpose

The `amalgemen` directory is the terminal stage of the UOR-FORGE pipeline. It receives the evaluated, certified student model from `forge/distill/` and `forge/eval/`, assembles the complete AMALGEMEN release artefact, and produces the semantic map — the structured document that makes AMALGEMEN the first semantically-addressed distilled model.

Nothing enters this directory without an eval certificate. Nothing leaves without a semantic map. These two constraints are the final gatekeepers of the forge.

## What This Stage Does

The `amalgemen` stage performs three operations:

| Operation | Description |
|---|---|
| **Release assembly** | Collect and validate all required artefacts: model weights, certificates, provenance index, alignment report |
| **Semantic map generation** | Construct the human- and machine-readable semantic map from the provenance index, coverage map, and alignment report |
| **Release packaging** | Package the model weights and semantic map into a versioned, distributable release |

## Inputs

| Input | Format | Source |
|---|---|---|
| Final model weights | `checkpoints/final/` | `forge/distill/` |
| Distillation certificate | `distill-cert.json` | `forge/distill/` |
| Coordinate alignment report | `alignment.md` | `forge/distill/` |
| Eval certificate | `eval-cert.json` | `forge/eval/` |
| Capability report | `capability.md` | `forge/eval/` |
| Semantic integrity report | `integrity.md` | `forge/eval/` |
| Provenance index | `provenance.json` | `forge/merge/` |
| Fusion certificates | `fusion-certs/` | `forge/merge/` |
| Donor contribution records | `{donor-id}/contribution.md` | `forge/donors/` |

## Outputs

| Output | Format | Description |
|---|---|---|
| AMALGEMEN weights | `release/amalgemen-{version}/weights/` | Final model weights, ready for deployment |
| Semantic map | `release/amalgemen-{version}/semantic-map.json` | Machine-readable map of all capabilities, coordinates, and provenance |
| Semantic map (human) | `release/amalgemen-{version}/semantic-map.md` | Human-readable version of the semantic map |
| Certificate chain | `release/amalgemen-{version}/certs/` | Complete chain: extraction → fusion → distillation → eval |
| Release manifest | `release/amalgemen-{version}/manifest.json` | Version, hash, and integrity metadata for the release |
| Release notes | `release/amalgemen-{version}/RELEASE.md` | Human-readable summary of the release |

## The Semantic Map

The semantic map is the defining artefact of AMALGEMEN and the primary output that distinguishes UOR-FORGE from all other distillation or merging approaches. It is a structured document that answers, for every capability in AMALGEMEN:

- **What** is this capability? (described in terms of the UOR coordinate it occupies)
- **Where** did it come from? (which teacher donor(s) contributed the source vectors)
- **How** was it fused? (which HARMONIA-DSL operators were applied)
- **How well** does the student represent it? (coordinate alignment score)
- **Is it certified?** (certificate chain reference)

### Semantic Map Schema (`semantic-map.json`)

```json
{
  "version": "1.0.0",
  "amalgemen_version": "amalgemen-7b-v1.0.0",
  "coordinate_substrate": "cat9-v6.3.0",
  "generated_at": "2026-03-11T00:00:00Z",
  "capabilities": [
    {
      "capability_id": "cap-0001",
      "coordinate": "https://uor.foundation/...",
      "description": "Human-readable description of the reasoning capability",
      "source_teachers": ["teacher-alpha-v1", "teacher-beta-v2"],
      "fusion_operator": "semantic-weighted-mean",
      "alignment_score": 0.91,
      "partition_class": "irreducible",
      "cert_chain": ["cert-extract-...", "cert-fusion-...", "cert-distill-...", "cert-eval-..."]
    }
  ]
}
```

## Release Versioning

AMALGEMEN releases follow semantic versioning (`MAJOR.MINOR.PATCH`) with the following conventions:

| Version Component | Meaning |
|---|---|
| `MAJOR` | Incompatible change to the coordinate substrate (cat9 topology version change) |
| `MINOR` | New teacher donors added or significant capability expansion |
| `PATCH` | Bug fixes, alignment improvements, or certificate chain corrections |

The cat9 topology version is always recorded in the release manifest. A release built against a different topology version is a different major version.

## Release Checklist

Before a release is finalised, all of the following must be verified:

- [ ] Eval certificate (`eval-cert.json`) is present and valid
- [ ] Distillation certificate (`distill-cert.json`) is present and valid
- [ ] Semantic map covers 100% of UOR coordinate regions in the fused vector set
- [ ] All donor contribution records are populated
- [ ] Certificate chain is complete from extraction through to eval
- [ ] Release manifest hash matches the model weights
- [ ] RELEASE.md is written and reviewed

## Invariants

1. No release may proceed without a valid eval certificate from `forge/eval/`.
2. The semantic map must achieve 100% coordinate coverage. A partial semantic map is not a valid release artefact.
3. The certificate chain must be complete. Any gap in the chain invalidates the release.
4. The release manifest must include the cat9 topology version used during distillation.

## Relationship to Other Stages

```
forge/distill/ ──── weights, distill-cert.json, alignment.md ──────────► forge/amalgemen/
forge/eval/    ──── eval-cert.json, capability.md, integrity.md ────────► forge/amalgemen/
forge/merge/   ──── provenance.json, fusion-certs/ ─────────────────────► forge/amalgemen/
forge/donors/  ──── contribution records ───────────────────────────────► forge/amalgemen/
                                                                               │
                                                                               └── release/amalgemen-{version}/
                                                                                       ├── weights/
                                                                                       ├── semantic-map.json
                                                                                       ├── semantic-map.md
                                                                                       ├── certs/
                                                                                       ├── manifest.json
                                                                                       └── RELEASE.md
```

---

*Stage 7 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
