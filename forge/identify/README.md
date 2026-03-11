# forge/identify — Stage 1: Teacher Model Identification

## Purpose

The `identify` stage is the entry point of the UOR-FORGE pipeline. Its responsibility is to enumerate, profile, and formally register the frontier teacher models that will contribute reasoning vectors to AMALGEMEN. No extraction, merging, or distillation occurs here. This stage exists solely to establish a verified, structured record of what teachers are available, what reasoning domains they cover, and whether they are compatible with the forge's coordinate substrate.

Identification is not a trivial step. The semantic map that ships with AMALGEMEN traces every capability back to a specific teacher at a specific version. That provenance chain begins here. A teacher that is not formally registered in this stage cannot contribute to any downstream stage.

---

## Stage 1a — Activation Patching: System 1 / System 2 Mode-Switching

The first identification experiment targets the **System 1 / System 2 mode-switching circuit** in `DeepSeek-R1-Distill-Llama-8B`. This is the mechanism by which the model decides to emit a `<think>` token and enter deliberate chain-of-thought reasoning, rather than producing a fast direct answer.

### Script: `identify_deepseek.py`

Uses **causal activation patching** via TransformerLens to localise the attention heads and MLP layers that carry the mode-switching signal.

**Methodology.** For each model component (attention head output, MLP layer output):

1. Run a *clean* forward pass on a System-2 prompt (reasoning benchmark).
2. Run a *corrupted* forward pass on a System-1 prompt (direct-answer).
3. Patch the clean activation of the component into the corrupted run.
4. Measure the **patch effect** as the change in log-probability of `<think>` at the first output position.

A high positive patch effect means the component carries information that distinguishes System-2 mode from System-1 mode — it is a **mode-switching component**.

### Architecture (DeepSeek-R1-Distill-Llama-8B)

| Parameter | Value |
|---|---|
| `model_type` | `llama` (LlamaForCausalLM) |
| `num_hidden_layers` | 32 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `hidden_size` | 4096 |
| `intermediate_size` | 14336 |
| `vocab_size` | 128256 |
| `max_position_embeddings` | 131072 |

### Prompt Pairs

| System 2 — clean (reasoning) | System 1 — corrupted (direct) |
|---|---|
| Modular arithmetic (2^100 mod 7) | Simple addition |
| Syllogistic proof (bloops/razzles) | Yes/no factual |
| Physics word problem (distance) | Simple multiplication |
| Free-fall kinematics | Colour recall |
| Proof: sum of two odd numbers | Name a fruit |
| Bubble sort trace | Capital city lookup |
| Counterfactual (boiling point) | Sensory description |
| Proof: sqrt(2) irrational | Simple division |

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Dry run — no model download, synthetic activations (for CI/testing)
python identify_deepseek.py --dry-run

# Full run on GPU (requires ~16 GB VRAM)
python identify_deepseek.py --device cuda --dtype bfloat16

# Partial run — layers 8-16 only, for fast iteration
python identify_deepseek.py --device cuda --dtype bfloat16 --layers 8-16

# Use a local model path
python identify_deepseek.py --model-path /data/models/DeepSeek-R1-Distill-Llama-8B
```

---

## What This Stage Does

The `identify` stage performs four operations in sequence:

| Operation | Description |
|---|---|
| **Enumeration** | List all candidate teacher models with their source, version, parameter count, and architecture family |
| **Domain profiling** | Map each teacher's known reasoning strengths to UOR coordinate regions in the cat9 topology |
| **Compatibility check** | Verify that the teacher's architecture and tokenisation are compatible with the forge's extraction protocol |
| **Registration** | Write a signed teacher manifest entry that downstream stages can reference by ID |

---

## Inputs

| Input | Format | Source |
|---|---|---|
| Teacher model weights | HuggingFace safetensors | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Prompt pairs | Hardcoded in `identify_deepseek.py` | `PROMPT_PAIRS` constant |
| cat9 topology coordinate map | Rust-generated from `foundation/` | `cargo run --bin uor-build` |
| Architecture compatibility matrix | JSON | Maintained in this directory |

---

## Outputs

| Output | Format | Consumed By |
|---|---|---|
| `layer_head_map_deepseek.json` | JSON (schema: `schema_layer_head_map.json`) | `forge/extract/` |
| `teachers.json` | JSON — one entry per registered teacher | `forge/extract/`, `forge/donors/` |
| `coverage.json` | JSON — UOR coordinate regions per teacher | `forge/merge/`, `forge/eval/` |
| `compat.md` | Markdown — human-readable pass/fail per teacher | Review and audit |

### `layer_head_map_deepseek.json` — Key Fields

| Field | Type | Description |
|---|---|---|
| `model_id` | string | HuggingFace model id |
| `think_token_id` | int | Vocabulary index of `<think>` |
| `patching_mode` | string | `"activation_patching"` or `"dry_run"` |
| `top_attention_heads` | array | Top 20 heads by mean patch effect |
| `top_mlp_layers` | array | Top 10 MLP layers by mean patch effect |
| `all_components` | array | Full ranked list of all components |

Each component record:

| Field | Description |
|---|---|
| `layer` | Zero-indexed transformer layer |
| `component_type` | `"attn_head"` or `"mlp"` |
| `head` | Head index (null for MLP) |
| `patch_effect_mean` | Mean patch effect across all prompt pairs |
| `patch_effect_std` | Standard deviation across prompt pairs |
| `patch_effects` | Per-pair raw scores |
| `rank` | Global rank (1 = highest mode-switching signal) |

### Teacher Manifest Schema (`teachers.json`)

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique identifier (e.g., `teacher-deepseek-r1-distill-8b-v1`) |
| `source` | string | Model source or organisation |
| `version` | string | Exact model version or checkpoint hash |
| `parameters` | integer | Parameter count |
| `architecture` | string | Architecture family (e.g., `llama`) |
| `domains` | array | UOR coordinate regions this teacher covers |
| `compatible` | boolean | Whether the teacher passed the compatibility check |
| `registered_at` | ISO 8601 | Timestamp of registration |

---

## Invariants

1. `patching_mode` must be `"activation_patching"` (not `"dry_run"`) before `layer_head_map_deepseek.json` is consumed by `forge/extract/`.
2. Every component in `all_components` must have a `rank` assigned.
3. `think_token_id` must resolve to a single token in the model vocabulary.
4. `num_prompt_pairs` must be ≥ 4 for statistical reliability.
5. The output file must validate against `schema_layer_head_map.json`.
6. A teacher may only proceed to `forge/extract/` if `compatible` is `true` in `teachers.json`.

---

## Data Flow

```
deepseek-ai/DeepSeek-R1-Distill-Llama-8B (HuggingFace)
        │
        ▼
identify_deepseek.py  (TransformerLens activation patching)
        │
        ▼
layer_head_map_deepseek.json
        │
        ▼
forge/extract/  ──► forge/merge/  ──► forge/distill/  ──► forge/amalgemen/
```

---

## Files

| File | Description |
|---|---|
| `identify_deepseek.py` | Stage 1a activation patching script |
| `layer_head_map_deepseek.json` | Output artefact (dry-run placeholder until live GPU run) |
| `schema_layer_head_map.json` | JSON Schema for the output artefact |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

*Stage 1 of 7 in the UOR-FORGE distillation pipeline. See [FORGE.md](../../FORGE.md) for the full pipeline specification.*
