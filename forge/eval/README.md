# forge/eval — Evaluation and Validation

## Purpose

The `eval` stage is the quality gate of the UOR-FORGE pipeline. It validates AMALGEMEN against two independent criteria: standard capability benchmarks and UOR-specific semantic integrity checks. A model that passes standard benchmarks but fails semantic integrity checks is not a valid AMALGEMEN release. Both criteria must be satisfied.

This dual-track evaluation reflects the dual nature of the project. AMALGEMEN must perform well as a language model — it must be useful. But it must also be semantically coherent — its capabilities must be traceable, its provenance complete, and its coordinate alignment verified. The `eval` stage enforces both requirements before the model is passed to `forge/amalgemen/` for final assembly.

## What This Stage Does

The `eval` stage performs two parallel tracks of evaluation:

### Track A — Capability Evaluation

| Check | Description |
|---|---|
| **Benchmark suite** | Run AMALGEMEN against standard reasoning, instruction-following, and domain-specific benchmarks |
| **Teacher comparison** | Compare AMALGEMEN's performance against each registered teacher on their profiled domains |
| **Regression check** | Verify that AMALGEMEN does not underperform relative to any single teacher on its primary domain |

### Track B — Semantic Integrity Evaluation

| Check | Description |
|---|---|
| **Coordinate coverage** | Verify that every UOR coordinate region in the fused vector set has a corresponding capability in AMALGEMEN |
| **Provenance completeness** | Verify that every capability in AMALGEMEN traces back to a source teacher via the provenance index |
| **Certificate chain integrity** | Verify that every fusion certificate and distillation certificate is valid and unbroken |
| **Alignment score threshold** | Verify that per-coordinate alignment scores from `forge/distill/` meet the minimum threshold |

## Inputs

| Input | Format | Source |
|---|---|---|
| Student model weights | `checkpoints/final/` | `forge/distill/` |
| Coordinate alignment report | `alignment.md` | `forge/distill/` |
| Distillation certificate | `distill-cert.json` | `forge/distill/` |
| Fusion certificates | `fusion-certs/` | `forge/merge/` |
| Provenance index | `provenance.json` | `forge/merge/` |
| Domain coverage map | `coverage.json` | `forge/identify/` |
| Extraction reports | `report/{teacher-id}.md` | `forge/extract/` |
| Eval config | `eval.toml` — benchmark selection and pass thresholds | Maintained in this directory |

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| Capability report | `capability.md` — benchmark scores, teacher comparisons, regression results | `forge/amalgemen/`, review |
| Semantic integrity report | `integrity.md` — coordinate coverage, provenance completeness, certificate chain status | `forge/amalgemen/`, review |
| Eval certificate | `eval-cert.json` — attests that both tracks passed; required for release | `forge/amalgemen/` |
| Failure log | `failures.md` — detailed failure records if either track does not pass | Remediation |

## Eval Config

`eval.toml` controls the evaluation parameters:

```toml
[capability]
benchmarks = ["mmlu", "hellaswag", "arc-challenge", "gsm8k", "humaneval"]
teacher_comparison = true
regression_tolerance = 0.02   # AMALGEMEN may underperform a teacher by at most 2% on its primary domain

[integrity]
require_full_coordinate_coverage = true
require_complete_provenance = true
min_alignment_score = 0.80
certificate_chain_depth = "full"   # validate every certificate in the chain, not just the leaf

[output]
fail_fast = false   # run all checks even if one fails; produce a complete failure log
```

## Pass Criteria

AMALGEMEN passes `eval` if and only if all of the following hold:

| Criterion | Track | Condition |
|---|---|---|
| Benchmark floor | A | All selected benchmarks meet or exceed configured minimum scores |
| No regression | A | AMALGEMEN does not underperform any teacher on its primary domain beyond the tolerance |
| Full coordinate coverage | B | Every UOR coordinate region in the fused vector set has a verified capability |
| Complete provenance | B | Every capability traces to a source teacher in the provenance index |
| Valid certificate chain | B | Every fusion and distillation certificate is valid |
| Alignment threshold | B | All per-coordinate alignment scores meet or exceed `min_alignment_score` |

If any criterion fails, the eval certificate is not issued and the model does not proceed to `forge/amalgemen/`.

## Invariants

1. The eval certificate is the only artefact that authorises passage to `forge/amalgemen/`. No certificate, no release.
2. Both tracks must pass independently. A Track A pass cannot compensate for a Track B failure, and vice versa.
3. The failure log must be produced whenever any check fails, with sufficient detail to guide remediation.

## Relationship to Other Stages

```
forge/distill/  ──── checkpoints/, alignment.md, distill-cert.json ──────► forge/eval/
forge/merge/    ──── fusion-certs/, provenance.json ──────────────────────► forge/eval/
forge/identify/ ──── coverage.json ───────────────────────────────────────► forge/eval/
forge/extract/  ──── report/ ─────────────────────────────────────────────► forge/eval/
                                                                                │
                                                                                ├── capability.md  ──► forge/amalgemen/
                                                                                ├── integrity.md   ──► forge/amalgemen/
                                                                                └── eval-cert.json ──► forge/amalgemen/
```

---

*Stage 5 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
