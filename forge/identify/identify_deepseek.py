"""
identify_deepseek.py
====================
Stage 1 — Teacher Model Identification
UOR-FORGE | forge/identify/

Activation patching experiment on DeepSeek-R1-Distill-Llama-8B to identify
the attention heads and MLP layers responsible for System 1 / System 2
mode-switching behaviour.

System 1  — fast, pattern-matching, direct-answer mode (no <think> token)
System 2  — slow, deliberate, chain-of-thought reasoning mode (emits <think>)

Methodology
-----------
We use *causal activation patching* (Vig et al. 2020; Meng et al. 2022) via
TransformerLens.  For each component (attention head output, MLP layer output)
we:

  1. Run a *clean* forward pass on a System-2 prompt (reasoning benchmark).
  2. Run a *corrupted* forward pass on a System-1 prompt (direct-answer).
  3. Patch the clean activation of component C into the corrupted run.
  4. Measure the *patch effect* as the change in the log-probability of the
     System-2 trigger token (<think>) at the first output position.

A high positive patch effect means component C carries information that
distinguishes System-2 mode from System-1 mode — it is a *mode-switching
component*.

Architecture (DeepSeek-R1-Distill-Llama-8B)
--------------------------------------------
  model_type        : llama (LlamaForCausalLM)
  num_hidden_layers : 32
  num_attention_heads: 32  (Q heads)
  num_key_value_heads: 8   (KV heads — GQA with repeat_kv=4)
  hidden_size       : 4096
  intermediate_size : 14336
  vocab_size        : 128256
  max_pos_embeddings: 131072

GQA and hook_z
--------------
DeepSeek-R1-Distill-Llama-8B uses Grouped Query Attention (GQA) with 32 query
heads and 8 KV heads (repeat factor = 4).  TransformerLens loads this model
using GroupedQueryAttention (triggered when cfg.n_key_value_heads is not None
and differs from cfg.n_heads).  In GroupedQueryAttention.calculate_z_scores,
the V tensor is expanded from [batch, pos, 8, d_head] to [batch, pos, 32,
d_head] via torch.repeat_interleave BEFORE hook_z is fired.  Consequently:

  hook_z shape: [batch, pos, 32, d_head]   — full Q-head dimension

Per-head patching with value[:, :, head_idx, :] is therefore correct for all
head indices 0..31.  The 8 KV-head structure is internal to the attention
computation and is already resolved by the time hook_z is reached.

This is only true when cfg.ungroup_grouped_query_attention=False (the default).
If that flag were True, hook_z would have only 8 heads and the patching loop
would need to iterate over range(n_kv_heads) instead.  The validate_gqa_config()
function below asserts this invariant at runtime.

Output
------
  layer_head_map_deepseek.json  — ranked map of mode-switching components

Usage
-----
  # Full run (requires ~16 GB VRAM or CPU with patience):
  python identify_deepseek.py

  # Dry-run (no model download, uses synthetic random activations):
  python identify_deepseek.py --dry-run

  # Specify device and precision:
  python identify_deepseek.py --device cuda --dtype bfloat16

  # Limit layers for fast iteration:
  python identify_deepseek.py --layers 0-7

  # Use a local model path:
  python identify_deepseek.py --model-path /path/to/DeepSeek-R1-Distill-Llama-8B

Dependencies
------------
  pip install transformer-lens torch transformers einops jaxtyping
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("identify_deepseek")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# DeepSeek-R1 uses <think> to enter System-2 reasoning mode.
# The model emits this token when it decides to reason step-by-step.
THINK_TOKEN = "<think>"
THINK_TOKEN_ALT = "think"   # fallback if tokeniser doesn't have the full tag

# Architecture constants (from config.json — used only for dry-run and CLI defaults)
# The live patching path derives these from the loaded model's cfg at runtime.
NUM_LAYERS = 32
NUM_HEADS  = 32   # Q heads; KV heads = 8 under GQA (resolved to 32 in hook_z)

# Output file
OUTPUT_FILE = Path(__file__).parent / "layer_head_map_deepseek.json"

# ---------------------------------------------------------------------------
# Benchmark prompt pairs
# ---------------------------------------------------------------------------
# Each pair: (system2_prompt, system1_prompt)
# system2_prompt  — should trigger slow reasoning (System 2)
# system1_prompt  — should elicit a fast direct answer (System 1)
#
# The prompts are minimal and contrastive so that the *only* difference
# between the two runs is the reasoning mode trigger.

PROMPT_PAIRS: List[Tuple[str, str]] = [
    # Mathematical reasoning
    (
        "Solve step by step: What is the remainder when 2^100 is divided by 7?",
        "What is 2 + 2?",
    ),
    # Logical deduction
    (
        "Think carefully: If all bloops are razzles and all razzles are lazzles, "
        "are all bloops lazzles? Prove it.",
        "Is the sky blue?",
    ),
    # Multi-step arithmetic
    (
        "Reason through this: A train travels 60 mph for 2.5 hours, then 80 mph "
        "for 1.5 hours. What is the total distance?",
        "What is 5 times 6?",
    ),
    # Causal reasoning
    (
        "Work through the causal chain: A ball is dropped from 100m. "
        "Ignoring air resistance, how long does it take to hit the ground?",
        "What colour is grass?",
    ),
    # Proof-style
    (
        "Prove or disprove: The sum of two odd numbers is always even.",
        "Name a fruit.",
    ),
    # Algorithmic thinking
    (
        "Trace through this algorithm step by step: "
        "Bubble sort [5, 3, 8, 1, 2]. Show each pass.",
        "What is the capital of France?",
    ),
    # Counterfactual reasoning
    (
        "Reason carefully: If water boiled at 50°C instead of 100°C, "
        "what would be three consequences for cooking?",
        "What does water taste like?",
    ),
    # Mathematical proof
    (
        "Prove that sqrt(2) is irrational using proof by contradiction.",
        "What is 10 divided by 2?",
    ),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComponentScore:
    """Patch effect score for a single model component."""
    layer: int
    component_type: str          # "attn_head", "mlp"
    head: Optional[int]          # None for MLP
    patch_effect_mean: float     # mean across prompt pairs
    patch_effect_std: float      # std across prompt pairs
    patch_effects: List[float]   # per-pair scores
    rank: int = 0                # filled after sorting


@dataclass
class LayerHeadMap:
    """Full output artefact written to layer_head_map_deepseek.json."""
    model_id: str
    num_layers: int
    num_heads: int
    think_token: str
    think_token_id: int
    num_prompt_pairs: int
    patching_mode: str           # "activation_patching" | "dry_run"
    device: str
    dtype: str
    timestamp: str
    runtime_seconds: float
    top_attention_heads: List[Dict]
    top_mlp_layers: List[Dict]
    all_components: List[Dict]
    methodology: str = (
        "Causal activation patching. For each component, the clean activation "
        "(System-2 prompt) is patched into the corrupted run (System-1 prompt). "
        "Patch effect = change in log-prob of the <think> token at position 0 "
        "of the model output. Higher = stronger mode-switching role."
    )
    uor_stage: str = "forge/identify"
    uor_donor: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def validate_gqa_config(model) -> int:
    """
    Assert that the loaded model's GQA configuration is compatible with the
    per-head patching strategy used in this script, and return the number of
    heads that hook_z exposes.

    Under TransformerLens GroupedQueryAttention with
    ungroup_grouped_query_attention=False (the default), V is expanded from
    n_kv_heads to n_heads via repeat_interleave *before* hook_z is fired.
    Therefore hook_z always has shape [batch, pos, n_heads, d_head] where
    n_heads is the full query-head count (32 for DeepSeek-R1-Distill-8B).

    If ungroup_grouped_query_attention=True were set, hook_z would have only
    n_kv_heads (8) heads and the patching loop would need to be adjusted.
    This function raises an error in that case to prevent silent mis-patching.
    """
    cfg = model.cfg
    n_heads    = cfg.n_heads
    n_kv_heads = cfg.n_key_value_heads  # None means standard MHA (n_kv == n_q)
    ungroup    = getattr(cfg, "ungroup_grouped_query_attention", False)

    log.info(
        "GQA config: n_heads=%d, n_key_value_heads=%s, "
        "ungroup_grouped_query_attention=%s",
        n_heads, n_kv_heads, ungroup,
    )

    if n_kv_heads is not None and n_kv_heads != n_heads:
        # Model uses GQA
        if ungroup:
            raise RuntimeError(
                f"ungroup_grouped_query_attention=True is set on this model. "
                f"hook_z will have {n_kv_heads} heads (KV dimension), not "
                f"{n_heads} (Q dimension). The per-head patching loop must "
                f"iterate over range({n_kv_heads}) in this mode. "
                f"Either set ungroup_grouped_query_attention=False (default) "
                f"or update the patching loop accordingly."
            )
        repeat_kv = n_heads // n_kv_heads
        log.info(
            "GQA: %d KV heads expanded to %d Q heads in hook_z "
            "(repeat_kv=%d). Per-head patching iterates over range(%d). "
            "This is correct.",
            n_kv_heads, n_heads, repeat_kv, n_heads,
        )
    else:
        log.info("Standard MHA (no GQA): hook_z has %d heads.", n_heads)

    return n_heads


def load_model(
    model_path: str,
    device: str,
    dtype_str: str,
) -> Tuple["HookedTransformer", int, int]:
    """Load DeepSeek-R1-Distill-Llama-8B into TransformerLens.

    Returns (model, think_token_id, n_heads_for_patching).
    n_heads_for_patching is the head dimension of hook_z — always the full
    Q-head count (32) under GQA with ungroup_grouped_query_attention=False.
    """
    from transformer_lens import HookedTransformer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    log.info("Loading model: %s", model_path)
    log.info("Device: %s | dtype: %s", device, dtype_str)

    model = HookedTransformer.from_pretrained(
        model_path,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=dtype,
        device=device,
    )
    model.eval()

    # Validate GQA config and determine hook_z head dimension
    n_heads_for_patching = validate_gqa_config(model)

    # Resolve the <think> token id
    tokeniser = model.tokenizer
    think_id = _resolve_think_token_id(tokeniser)
    log.info("Think token id: %d", think_id)

    return model, think_id, n_heads_for_patching


def _resolve_think_token_id(tokeniser) -> int:
    """Find the token id for <think> in the DeepSeek tokeniser."""
    # Try the full tag first
    ids = tokeniser.encode(THINK_TOKEN, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Try without angle brackets
    ids = tokeniser.encode(THINK_TOKEN_ALT, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Fall back to the first subword of <think>
    log.warning(
        "Could not resolve <think> as a single token. "
        "Using first subword id: %d", ids[0]
    )
    return ids[0]


# ---------------------------------------------------------------------------
# Activation patching core
# ---------------------------------------------------------------------------

def get_logprob_think(
    logits: torch.Tensor,
    think_token_id: int,
    position: int = 0,
) -> float:
    """
    Extract the log-probability of the think token at the given output position.
    logits: [batch, seq, vocab]
    """
    # We look at position -1 (the last token's prediction = first output token)
    last_logits = logits[0, -1, :]          # [vocab]
    log_probs = F.log_softmax(last_logits, dim=-1)
    return log_probs[think_token_id].item()


def patch_effect_for_component(
    model,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    think_token_id: int,
    hook_name: str,
    head_idx: Optional[int],
    clean_cache: Dict,
) -> float:
    """
    Compute the patch effect for a single component.

    Patches the clean activation of `hook_name` (optionally sliced to
    `head_idx`) into a fresh corrupted forward pass, and returns the
    change in log-prob of the think token.
    """
    # Baseline: corrupted run without patching
    with torch.no_grad():
        corrupted_logits = model(corrupted_tokens)
    baseline_lp = get_logprob_think(corrupted_logits, think_token_id)

    # Patched run
    def patch_hook(value, hook):
        clean_act = clean_cache[hook_name]
        if head_idx is not None:
            # value shape: [batch, seq, n_heads, d_head]
            value[:, :, head_idx, :] = clean_act[:, :, head_idx, :]
        else:
            value[:] = clean_act
        return value

    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
    patched_lp = get_logprob_think(patched_logits, think_token_id)

    return patched_lp - baseline_lp


def run_activation_patching(
    model,
    think_token_id: int,
    layer_range: range,
    prompt_pairs: List[Tuple[str, str]],
    n_heads: int,
) -> List[ComponentScore]:
    """
    Run the full activation patching experiment across all components
    in `layer_range` for all prompt pairs.

    n_heads is the head dimension of hook_z as returned by validate_gqa_config().
    For DeepSeek-R1-Distill-8B under GQA with ungroup=False this is 32 (the full
    Q-head count), not 8 (the KV-head count).

    Returns a list of ComponentScore objects.
    """
    scores: Dict[Tuple, List[float]] = {}

    for pair_idx, (s2_prompt, s1_prompt) in enumerate(prompt_pairs):
        log.info(
            "Prompt pair %d/%d: '%s...' vs '%s...'",
            pair_idx + 1, len(prompt_pairs),
            s2_prompt[:40], s1_prompt[:40],
        )

        # Tokenise
        clean_tokens = model.to_tokens(s2_prompt)       # System 2 = clean
        corrupted_tokens = model.to_tokens(s1_prompt)   # System 1 = corrupted

        # Cache clean activations
        with torch.no_grad():
            _, clean_cache = model.run_with_cache(clean_tokens)

        for layer in layer_range:
            # --- Attention heads ---
            # hook_z shape: [batch, seq, n_heads, d_head]
            # Under GQA (ungroup=False), n_heads = Q-head count (32), NOT KV-head
            # count (8). V is expanded inside calculate_z_scores before hook_z fires.
            attn_hook = f"blocks.{layer}.attn.hook_z"
            for head in range(n_heads):
                key = (layer, "attn_head", head)
                effect = patch_effect_for_component(
                    model, clean_tokens, corrupted_tokens,
                    think_token_id, attn_hook, head, clean_cache,
                )
                scores.setdefault(key, []).append(effect)

            # --- MLP layer ---
            mlp_hook = f"blocks.{layer}.hook_mlp_out"   # [batch, seq, d_model]
            key = (layer, "mlp", None)
            effect = patch_effect_for_component(
                model, clean_tokens, corrupted_tokens,
                think_token_id, mlp_hook, None, clean_cache,
            )
            scores.setdefault(key, []).append(effect)

            log.info("  Layer %02d complete", layer)

    # Build ComponentScore objects
    components: List[ComponentScore] = []
    for (layer, ctype, head), effects in scores.items():
        t = torch.tensor(effects, dtype=torch.float32)
        components.append(ComponentScore(
            layer=layer,
            component_type=ctype,
            head=head,
            patch_effect_mean=t.mean().item(),
            patch_effect_std=t.std().item() if len(effects) > 1 else 0.0,
            patch_effects=effects,
        ))

    # Rank by mean patch effect (descending)
    components.sort(key=lambda c: c.patch_effect_mean, reverse=True)
    for rank, comp in enumerate(components, start=1):
        comp.rank = rank

    return components


# ---------------------------------------------------------------------------
# Dry-run (synthetic) mode
# ---------------------------------------------------------------------------

def run_dry_run(layer_range: range) -> List[ComponentScore]:
    """
    Generate synthetic patch-effect scores for testing without a GPU.
    Uses a realistic distribution: a few high-signal heads in layers 8-16,
    with noise elsewhere, mimicking known attention-head localisation patterns.
    """
    import random
    random.seed(42)
    torch.manual_seed(42)

    components: List[ComponentScore] = []
    # Simulate high-signal heads concentrated in middle layers
    high_signal_heads = {
        (10, 7), (11, 3), (12, 15), (13, 22), (14, 8),
        (15, 1), (16, 19), (9, 11), (17, 4), (10, 21),
    }
    high_signal_mlp = {10, 11, 12, 13, 14, 15}

    for layer in layer_range:
        for head in range(NUM_HEADS):
            if (layer, head) in high_signal_heads:
                base = random.uniform(0.35, 0.65)
            elif layer in range(8, 20):
                base = random.uniform(0.05, 0.20)
            else:
                base = random.uniform(-0.05, 0.08)
            effects = [base + random.gauss(0, 0.03) for _ in range(len(PROMPT_PAIRS))]
            t = torch.tensor(effects)
            components.append(ComponentScore(
                layer=layer,
                component_type="attn_head",
                head=head,
                patch_effect_mean=t.mean().item(),
                patch_effect_std=t.std().item(),
                patch_effects=effects,
            ))

        # MLP
        if layer in high_signal_mlp:
            base = random.uniform(0.20, 0.45)
        else:
            base = random.uniform(-0.03, 0.10)
        effects = [base + random.gauss(0, 0.02) for _ in range(len(PROMPT_PAIRS))]
        t = torch.tensor(effects)
        components.append(ComponentScore(
            layer=layer,
            component_type="mlp",
            head=None,
            patch_effect_mean=t.mean().item(),
            patch_effect_std=t.std().item(),
            patch_effects=effects,
        ))

    components.sort(key=lambda c: c.patch_effect_mean, reverse=True)
    for rank, comp in enumerate(components, start=1):
        comp.rank = rank
    return components


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def build_output(
    components: List[ComponentScore],
    model_id: str,
    think_token_id: int,
    num_prompt_pairs: int,
    patching_mode: str,
    device: str,
    dtype: str,
    runtime_seconds: float,
    layer_range: range,
) -> LayerHeadMap:
    """Assemble the LayerHeadMap output artefact."""
    import datetime

    attn_components = [c for c in components if c.component_type == "attn_head"]
    mlp_components  = [c for c in components if c.component_type == "mlp"]

    def _serialise(c: ComponentScore) -> Dict:
        d = asdict(c)
        # Round floats for readability
        d["patch_effect_mean"] = round(d["patch_effect_mean"], 6)
        d["patch_effect_std"]  = round(d["patch_effect_std"],  6)
        d["patch_effects"]     = [round(x, 6) for x in d["patch_effects"]]
        return d

    return LayerHeadMap(
        model_id=model_id,
        num_layers=len(layer_range),
        num_heads=NUM_HEADS,
        think_token=THINK_TOKEN,
        think_token_id=think_token_id,
        num_prompt_pairs=num_prompt_pairs,
        patching_mode=patching_mode,
        device=device,
        dtype=dtype,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        runtime_seconds=round(runtime_seconds, 2),
        top_attention_heads=[_serialise(c) for c in attn_components[:20]],
        top_mlp_layers=[_serialise(c) for c in mlp_components[:10]],
        all_components=[_serialise(c) for c in components],
    )


def write_output(layer_head_map: LayerHeadMap, output_path: Path) -> None:
    """Write the LayerHeadMap to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(layer_head_map), f, indent=2)
    log.info("Output written to: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 — Activation patching on DeepSeek-R1-Distill-8B "
                    "to identify System 1/2 mode-switching components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-path",
        default=MODEL_ID,
        help=f"HuggingFace model id or local path (default: {MODEL_ID})",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
        help="Compute device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    p.add_argument(
        "--layers",
        default=f"0-{NUM_LAYERS - 1}",
        help="Layer range to patch, e.g. '0-31' or '8-16' (default: all layers)",
    )
    p.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help=f"Output JSON path (default: {OUTPUT_FILE})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with synthetic activations (no model download required)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top components to highlight in logs (default: 20)",
    )
    return p.parse_args()


def parse_layer_range(spec: str) -> range:
    """Parse '0-31' or '8-16' into a range."""
    parts = spec.split("-")
    if len(parts) == 2:
        return range(int(parts[0]), int(parts[1]) + 1)
    if len(parts) == 1:
        n = int(parts[0])
        return range(n, n + 1)
    raise ValueError(f"Invalid layer range: {spec!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    layer_range = parse_layer_range(args.layers)
    output_path = Path(args.output)

    log.info("=" * 60)
    log.info("UOR-FORGE Stage 1 — Identify: DeepSeek-R1-Distill-8B")
    log.info("Target behaviour: System 1 / System 2 mode-switching")
    n_heads_display = NUM_HEADS if args.dry_run else "(from model cfg)"
    log.info("Layers: %d-%d | Heads: %s | Pairs: %d",
             layer_range.start, layer_range.stop - 1,
             n_heads_display, len(PROMPT_PAIRS))
    log.info("=" * 60)

    t0 = time.time()

    if args.dry_run:
        log.info("DRY RUN — using synthetic activations")
        components = run_dry_run(layer_range)
        think_token_id = 9999   # synthetic
        patching_mode = "dry_run"
        device = "cpu"
        dtype = "float32"
    else:
        model, think_token_id, n_heads_live = load_model(
            args.model_path, args.device, args.dtype
        )
        components = run_activation_patching(
            model, think_token_id, layer_range, PROMPT_PAIRS,
            n_heads=n_heads_live,
        )
        patching_mode = "activation_patching"
        device = args.device
        dtype = args.dtype

    runtime = time.time() - t0

    # Report top components
    log.info("\n--- Top %d Mode-Switching Components ---", args.top_n)
    for comp in components[:args.top_n]:
        head_str = f"head={comp.head:02d}" if comp.head is not None else "mlp    "
        log.info(
            "  Rank %03d | Layer %02d | %s | %s | "
            "mean_effect=%.4f ± %.4f",
            comp.rank, comp.layer, comp.component_type, head_str,
            comp.patch_effect_mean, comp.patch_effect_std,
        )

    # Build and write output
    layer_head_map = build_output(
        components=components,
        model_id=args.model_path,
        think_token_id=think_token_id,
        num_prompt_pairs=len(PROMPT_PAIRS),
        patching_mode=patching_mode,
        device=device,
        dtype=dtype,
        runtime_seconds=runtime,
        layer_range=layer_range,
    )
    write_output(layer_head_map, output_path)

    log.info("Done in %.1fs", runtime)


if __name__ == "__main__":
    main()
