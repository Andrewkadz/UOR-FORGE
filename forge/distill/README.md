# forge/distill — Student Model Distillation

## Purpose

The `distill` stage trains AMALGEMEN — the 7B student model — using the fused, semantically-addressed vector set produced by `forge/merge/`. This is where the abstract coordinate-space work of the earlier stages is translated into concrete model weights.

Distillation in UOR-FORGE is not classical knowledge distillation (soft-label transfer from teacher logits). It is **addressed capability transfer**: the student is trained to reproduce the reasoning behaviours encoded in the fused vector set, where each behaviour is anchored to a UOR coordinate. The training objective is not merely to minimise loss on teacher outputs — it is to ensure that the student's internal representations align with the semantic coordinates of the fused vectors.

The output of this stage is not just a set of model weights. It is a model weights file paired with a coordinate alignment report that documents how well the student's representations match the target UOR coordinates.

## What This Stage Does

The `distill` stage performs three operations:

| Operation | Description |
|---|---|
| **Training data construction** | Convert the fused vector set into a training corpus aligned to the student architecture |
| **Student training** | Train AMALGEMEN 7B using the constructed corpus, with coordinate alignment as an auxiliary objective |
| **Alignment verification** | Probe the trained student to verify that its internal representations align with the target UOR coordinates |

## Inputs

| Input | Format | Source |
|---|---|---|
| Fused vector set | `fused/` | `forge/merge/` |
| Fusion certificates | `fusion-certs/` | `forge/merge/` |
| Distillation config | `distill.toml` — training hyperparameters and alignment objectives | Maintained in this directory |
| Student architecture spec | `arch.json` — AMALGEMEN 7B architecture definition | `forge/amalgemen/` |

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| Student model weights | `checkpoints/` — training checkpoints + final weights | `forge/amalgemen/`, `forge/eval/` |
| Coordinate alignment report | `alignment.md` — per-coordinate alignment scores | `forge/eval/`, `forge/amalgemen/` |
| Training log | `train.log` — loss curves, alignment metrics, hardware stats | Audit and reproducibility |
| Distillation certificate | `distill-cert.json` — attests the training run against the fused vector set | `forge/amalgemen/` |

## Distillation Config

`distill.toml` controls the training process:

```toml
[model]
architecture = "amalgemen-7b"
checkpoint_base = ""              # optional: start from an existing checkpoint

[training]
epochs = 3
batch_size = 32
learning_rate = 2e-5
warmup_steps = 500
gradient_checkpointing = true

[alignment]
coordinate_loss_weight = 0.25     # weight of the UOR coordinate alignment loss
alignment_probe_interval = 500    # steps between alignment probes
alignment_threshold = 0.80        # minimum per-coordinate alignment score to pass

[output]
checkpoint_interval = 1000        # save a checkpoint every N steps
final_checkpoint = "final/"
```

## Coordinate Alignment Objective

In addition to the standard language modelling or instruction-following loss, the distillation training incorporates a **coordinate alignment loss** that penalises the student when its internal representations diverge from the UOR coordinates of the fused vectors it is trained on.

This auxiliary objective is what distinguishes AMALGEMEN from a conventionally distilled model. It ensures that the student does not merely learn to produce similar outputs — it learns to reason in a way that is geometrically consistent with the cat9 coordinate substrate.

The alignment probe runs at regular intervals during training, measuring the cosine similarity between the student's internal representations and the target UOR coordinates. Results are recorded in `alignment.md`.

## Invariants

1. The distillation certificate must be issued before the final checkpoint is passed to `forge/amalgemen/`. An uncertified checkpoint is not a valid AMALGEMEN candidate.
2. The coordinate alignment report must cover every UOR coordinate region present in the fused vector set. Gaps in alignment coverage are a stage failure.
3. The training log must be preserved alongside the final checkpoint for reproducibility.

## Relationship to Other Stages

```
forge/merge/
    │
    ├── fused/        ──────────────────► forge/distill/
    └── fusion-certs/ ──────────────────► forge/distill/
                                              │
                                              ├── checkpoints/      ──────► forge/amalgemen/
                                              │                     ──────► forge/eval/
                                              ├── alignment.md      ──────► forge/eval/
                                              │                     ──────► forge/amalgemen/
                                              └── distill-cert.json ──────► forge/amalgemen/
```

---

*Stage 4 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
