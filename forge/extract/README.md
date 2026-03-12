# forge/extract — Reasoning Vector Extraction

## Purpose

The `extract` stage draws high-signal reasoning vectors from each registered teacher model and assigns each vector a coordinate address in the cat9 topology. This is the first stage at which the UOR semantic layer becomes active. Raw weight tensors enter; addressed reasoning vectors leave.

Extraction is the most computationally intensive stage in the pipeline. It operates directly on teacher model weights and must produce outputs that are fully addressable — every vector that exits this stage carries a valid UOR coordinate, or it is discarded. Unaddressed vectors are not passed downstream under any circumstances (see FORGE.md, Invariant 2).

## Stage 2a — DARE Extraction: DeepSeek-R1-Distill-8B Mode-Switching Circuits

The first extraction experiment targets the mode-switching circuits identified in Stage 1 (`forge/identify/layer_head_map_deepseek.json`). It applies **targeted DARE** to extract sparse, circuit-localised delta tensors from `DeepSeek-R1-Distill-Llama-8B`.

### Script: `extract_deepseek.py`

Implements the full extraction pipeline: Stage 1 artefact validation → DARE sparsification → UOR metadata wrapping → safetensors serialisation.

### What is DARE?

**DARE (Drop And REscale)** (Yu et al., 2023) is a weight-delta sparsification method. Given a fine-tuned model and a base model, it:

1. Computes the delta: `Δ = θ_ft - θ_base`
2. Randomly drops (zeros) a fraction `p` of delta weights via a Bernoulli mask
3. Rescales surviving weights by `1/(1-p)` to preserve expected magnitude

In UOR-FORGE, DARE is applied in a **targeted** variant: only the weight matrices corresponding to the high-signal circuits from Stage 1 are processed. All other weight matrices are excluded entirely.

### Stage 1 Gate (Hard Exit)

The script reads `forge/identify/layer_head_map_deepseek.json` and enforces the following invariant before proceeding:

```
if _status == "PENDING_LIVE_RUN"  →  hard exit (sys.exit(1))
if patching_mode == "dry_run"     →  hard exit (sys.exit(1))
```

Stage 2 must not run on synthetic Stage 1 data.

### UOR Metadata Envelope

Each extracted tensor is wrapped with a `UORMetaRecord` containing:

| Field | Description |
|---|---|
| `tensor_key` | Stable key in the safetensors file (e.g., `L12_H15_delta`) |
| `donor` | Source teacher model identifier |
| `vector_target` | Behavioural target (`system1_system2_mode_switch`) |
| `layer` | Transformer layer index |
| `component_type` | `attn_head` or `mlp` |
| `head` | Head index (null for MLP) |
| `uor_address` | Placeholder UOR coordinate (see note below) |
| `patch_effect_mean` | Stage 1 patch effect score |
| `rank` | Stage 1 rank (1 = highest mode-switching signal) |
| `dare_p` | Drop probability applied |
| `dare_rescale` | Rescale factor `1/(1-p)` |
| `sparsity_achieved` | Actual fraction zeroed after DARE |
| `content_hash` | SHA-256 of tensor bytes for integrity |

> **Note on `uor_address`:** Fields are currently placeholder coordinates in the format `uor://forge.identify/deepseek-r1-distill-8b/L{layer}/{component}/H{head}`. Full resolution against the cat9 topology will be defined in a separate coordinate schema and implemented in `forge/merge/`.

### Usage

```bash
# Dry run — validates full pipeline logic without loading the model
python extract_deepseek.py --dry-run

# Full live run (requires ~32 GB VRAM for both models)
python extract_deepseek.py --device cuda --dtype bfloat16

# Custom DARE parameters
python extract_deepseek.py --device cuda --dare-p 0.85 --top-n 30
```

### Outputs

| Output | Format | Consumed By |
|---|---|---|
| `deepseek_delta_uor.safetensors` | safetensors | `forge/merge/` |
| `deepseek_delta_uor_meta.json` | JSON (schema: `schema_uor_meta.json`) | `forge/merge/`, `forge/eval/` |

### References

Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2023). **Language Model Arithmetic**. *arXiv:2311.03099*. https://arxiv.org/abs/2311.03099

---

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
