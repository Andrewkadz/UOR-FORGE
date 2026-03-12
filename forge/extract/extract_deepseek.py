"""
extract_deepseek.py
===================
Stage 2 — Reasoning Vector Extraction
UOR-FORGE | forge/extract/

Applies DARE (Drop And REscale) to the mode-switching circuits identified in
Stage 1, extracts the resulting delta tensors, and wraps each tensor with a
UOR metadata envelope before saving to safetensors format.

What is DARE?
-------------
DARE (Yu et al., 2023 — "Language Model Arithmetic") is a weight-delta
sparsification method. Given a fine-tuned model and a base model, it:

  1. Computes the delta: Δ = θ_ft - θ_base
  2. Randomly drops (zeros) a fraction p of delta weights
  3. Rescales the surviving weights by 1/(1-p) to preserve expected magnitude

In UOR-FORGE we apply a *targeted* variant: DARE is applied only to the
weight matrices corresponding to the high-signal circuits identified in
Stage 1 (specific attention heads and MLP layers). All other weight matrices
are zeroed entirely — they carry no signal of interest for the mode-switching
behaviour and should not contribute to the fused student.

This produces a sparse, circuit-localised delta tensor that is:
  - Semantically meaningful (anchored to identified mode-switching components)
  - Magnitude-normalised (DARE rescaling preserves expected signal strength)
  - Addressable (every tensor is tagged with a UOR metadata envelope)

UOR Metadata Envelope
---------------------
Each extracted tensor is wrapped with a UOR metadata record containing:

  donor          — source model identifier
  vector_target  — the behavioural target (e.g., "system1_system2_mode_switch")
  source_layers  — list of (layer, component_type, head) tuples that contributed
  uor_address    — placeholder UOR coordinate address (format: uor://<ns>/<path>)
                   Full coordinate resolution will be defined in a separate spec.
  dare_p         — drop probability used
  dare_rescale   — rescale factor applied (= 1/(1-p))
  signal_rank_threshold — minimum rank from Stage 1 included in extraction
  extraction_timestamp  — ISO 8601 UTC

Output
------
  deepseek_delta_uor.safetensors  — sparse delta tensors for target circuits
  deepseek_delta_uor_meta.json    — UOR metadata sidecar

Usage
-----
  # Full run (requires model weights and ~16 GB VRAM):
  python extract_deepseek.py

  # Dry-run (validates full pipeline logic without loading the model):
  python extract_deepseek.py --dry-run

  # Custom parameters:
  python extract_deepseek.py \\
      --model-path /data/models/DeepSeek-R1-Distill-Llama-8B \\
      --base-model-path /data/models/Llama-3.1-8B \\
      --dare-p 0.9 \\
      --top-n 20 \\
      --device cuda \\
      --dtype bfloat16

  # Use a specific layer_head_map:
  python extract_deepseek.py --layer-head-map /path/to/layer_head_map.json

Dependencies
------------
  pip install -r requirements.txt
  (transformer-lens, torch, transformers, safetensors, einops)

References
----------
  Yu et al. (2023). Language Model Arithmetic.
  https://arxiv.org/abs/2311.03099
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_deepseek")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

LAYER_HEAD_MAP_PATH = REPO_ROOT / "forge" / "identify" / "layer_head_map_deepseek.json"
OUTPUT_DIR          = Path(__file__).parent

SAFETENSORS_OUTPUT  = OUTPUT_DIR / "deepseek_delta_uor.safetensors"
META_OUTPUT         = OUTPUT_DIR / "deepseek_delta_uor_meta.json"

MODEL_ID      = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"   # base model for delta computation

# Architecture constants (DeepSeek-R1-Distill-Llama-8B)
NUM_LAYERS = 32
NUM_HEADS  = 32
D_HEAD     = 128   # hidden_size (4096) / num_attention_heads (32)
D_MODEL    = 4096
D_MLP      = 14336

# DARE default drop probability (Yu et al. recommend 0.9 for merging)
DEFAULT_DARE_P = 0.9

# UOR address namespace placeholder — will be replaced by the full coordinate spec
UOR_NS_PLACEHOLDER = "uor://forge.identify/deepseek-r1-distill-8b"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TargetComponent:
    """A single circuit component selected from the Stage 1 layer_head_map."""
    layer: int
    component_type: str   # "attn_head" | "mlp"
    head: Optional[int]   # None for MLP
    patch_effect_mean: float
    rank: int

    def weight_keys(self) -> List[str]:
        """
        Return the TransformerLens weight key(s) for this component.

        Attention head (hook_z path):
          W_O projects from [n_heads * d_head] → d_model.
          To isolate head h, we extract the slice of W_O corresponding to head h:
            blocks.{L}.attn.W_O  shape: [n_heads, d_head, d_model]
          and the corresponding input projection:
            blocks.{L}.attn.W_V  shape: [n_heads, d_model, d_head]  (or GQA variant)

        MLP layer:
            blocks.{L}.mlp.W_in   shape: [d_model, d_mlp]
            blocks.{L}.mlp.W_out  shape: [d_mlp, d_model]
            blocks.{L}.mlp.W_gate shape: [d_model, d_mlp]  (gated MLP)
        """
        L = self.layer
        if self.component_type == "attn_head":
            return [
                f"blocks.{L}.attn.W_O",   # output projection (sliced per head)
                f"blocks.{L}.attn.W_V",   # value projection (sliced per KV group)
            ]
        else:  # mlp
            return [
                f"blocks.{L}.mlp.W_in",
                f"blocks.{L}.mlp.W_out",
                f"blocks.{L}.mlp.W_gate",
            ]

    def tensor_key(self) -> str:
        """Stable key for the output safetensors tensor."""
        if self.component_type == "attn_head":
            return f"L{self.layer:02d}_H{self.head:02d}_delta"
        else:
            return f"L{self.layer:02d}_MLP_delta"

    def uor_address(self) -> str:
        """
        Placeholder UOR coordinate address for this component.

        Format: uor://<namespace>/<layer>/<component>/<head>
        The full coordinate resolution against the cat9 topology will be
        defined in a separate spec and implemented in forge/merge/.
        """
        if self.component_type == "attn_head":
            return f"{UOR_NS_PLACEHOLDER}/L{self.layer:02d}/attn/H{self.head:02d}"
        else:
            return f"{UOR_NS_PLACEHOLDER}/L{self.layer:02d}/mlp"


@dataclass
class UORMetaRecord:
    """UOR metadata envelope for a single extracted delta tensor."""
    tensor_key: str
    donor: str
    vector_target: str
    layer: int
    component_type: str
    head: Optional[int]
    source_layers: List[int]
    uor_address: str
    patch_effect_mean: float
    rank: int
    dare_p: float
    dare_rescale: float
    sparsity_achieved: float   # fraction of weights zeroed after DARE
    tensor_shape: List[int]
    tensor_dtype: str
    content_hash: str          # SHA-256 of the tensor bytes (for integrity)
    extraction_timestamp: str


@dataclass
class ExtractionManifest:
    """Full sidecar metadata written to deepseek_delta_uor_meta.json."""
    schema_version: str = "0.1.0"
    uor_stage: str = "forge/extract"
    uor_donor: str = MODEL_ID
    base_model: str = BASE_MODEL_ID
    vector_target: str = "system1_system2_mode_switch"
    patching_mode: str = "dare_targeted"
    device: str = "cpu"
    dtype: str = "float32"
    dare_p: float = DEFAULT_DARE_P
    dare_rescale: float = field(default_factory=lambda: 1.0 / (1.0 - DEFAULT_DARE_P))
    top_n_components: int = 20
    signal_rank_threshold: int = 20
    num_tensors: int = 0
    total_parameters: int = 0
    total_nonzero_parameters: int = 0
    overall_sparsity: float = 0.0
    layer_head_map_path: str = str(LAYER_HEAD_MAP_PATH)
    layer_head_map_status: str = ""
    safetensors_path: str = str(SAFETENSORS_OUTPUT)
    extraction_timestamp: str = ""
    runtime_seconds: float = 0.0
    tensors: List[Dict] = field(default_factory=list)
    uor_address_note: str = (
        "uor_address fields are placeholder coordinates. Full resolution "
        "against the cat9 topology will be implemented in forge/merge/ once "
        "the UOR coordinate schema is finalised."
    )


# ---------------------------------------------------------------------------
# Stage 1 artefact loading and validation
# ---------------------------------------------------------------------------

def load_and_validate_layer_head_map(path: Path) -> Dict:
    """
    Load layer_head_map_deepseek.json and enforce the LIVE status invariant.

    Exits with code 1 if _status == PENDING_LIVE_RUN or patching_mode == dry_run.
    This is a hard gate: Stage 2 must not proceed on synthetic Stage 1 data.
    """
    if not path.exists():
        log.error("layer_head_map not found: %s", path)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    status = data.get("_status", "")
    patching_mode = data.get("patching_mode", "")

    if status == "PENDING_LIVE_RUN":
        log.error(
            "HARD EXIT: layer_head_map_deepseek.json has _status=PENDING_LIVE_RUN.\n"
            "  Stage 2 requires a live activation-patching run from Stage 1.\n"
            "  Run: python forge/identify/identify_deepseek.py --device cuda --dtype bfloat16\n"
            "  Then re-run this script."
        )
        sys.exit(1)

    if patching_mode == "dry_run":
        log.error(
            "HARD EXIT: layer_head_map_deepseek.json has patching_mode=dry_run.\n"
            "  Stage 2 requires patching_mode=activation_patching.\n"
            "  Run Stage 1 with a live model before proceeding."
        )
        sys.exit(1)

    log.info(
        "Layer head map validated: model=%s, patching_mode=%s, "
        "num_pairs=%d, num_components=%d",
        data.get("model_id", "?"),
        patching_mode,
        data.get("num_prompt_pairs", 0),
        len(data.get("all_components", [])),
    )
    return data


def select_target_components(
    layer_head_map: Dict,
    top_n: int,
) -> List[TargetComponent]:
    """
    Select the top-N components from the layer_head_map by rank.

    Returns separate lists for attention heads and MLP layers, merged and
    sorted by rank. Only components with rank <= top_n are included.
    """
    all_components = layer_head_map.get("all_components", [])

    # If all_components is empty (dry-run placeholder), fall back to
    # top_attention_heads + top_mlp_layers
    if not all_components:
        all_components = (
            layer_head_map.get("top_attention_heads", []) +
            layer_head_map.get("top_mlp_layers", [])
        )
        log.warning(
            "all_components is empty — falling back to top_attention_heads "
            "+ top_mlp_layers (%d components)", len(all_components)
        )

    # Filter to top_n by rank
    targets = []
    for comp in all_components:
        rank = comp.get("rank", 9999)
        if rank <= top_n:
            targets.append(TargetComponent(
                layer=comp["layer"],
                component_type=comp["component_type"],
                head=comp.get("head"),
                patch_effect_mean=comp.get("patch_effect_mean", 0.0),
                rank=rank,
            ))

    targets.sort(key=lambda c: c.rank)
    log.info("Selected %d target components (top_n=%d)", len(targets), top_n)
    for t in targets[:5]:
        log.info(
            "  rank=%d layer=%02d type=%s head=%s effect=%.4f",
            t.rank, t.layer, t.component_type,
            str(t.head) if t.head is not None else "mlp",
            t.patch_effect_mean,
        )
    if len(targets) > 5:
        log.info("  ... and %d more", len(targets) - 5)
    return targets


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    model_path: str,
    base_model_path: str,
    device: str,
    dtype_str: str,
) -> Tuple:
    """
    Load the fine-tuned model and the base model into TransformerLens.

    Returns (ft_model, base_model).
    Both models are loaded in eval mode with the same dtype and device.
    """
    from transformer_lens import HookedTransformer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    log.info("Loading fine-tuned model: %s", model_path)
    ft_model = HookedTransformer.from_pretrained(
        model_path,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=dtype,
        device=device,
    )
    ft_model.eval()
    log.info("Fine-tuned model loaded: %d layers, %d heads",
             ft_model.cfg.n_layers, ft_model.cfg.n_heads)

    log.info("Loading base model: %s", base_model_path)
    base_model = HookedTransformer.from_pretrained(
        base_model_path,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=dtype,
        device=device,
    )
    base_model.eval()
    log.info("Base model loaded.")

    return ft_model, base_model


# ---------------------------------------------------------------------------
# DARE implementation
# ---------------------------------------------------------------------------

def compute_delta(
    ft_weight: torch.Tensor,
    base_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the weight delta: Δ = θ_ft - θ_base.

    Both tensors must have the same shape. The result is returned in float32
    regardless of input dtype to preserve precision during sparsification.
    """
    return ft_weight.float() - base_weight.float()


def dare_sparsify(
    delta: torch.Tensor,
    p: float,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Apply DARE sparsification to a delta tensor.

    Algorithm (Yu et al. 2023):
      1. Sample a Bernoulli mask M ~ Bernoulli(1-p) — each weight survives
         with probability (1-p).
      2. Zero dropped weights: Δ_sparse = Δ * M
      3. Rescale surviving weights: Δ_dare = Δ_sparse / (1-p)

    Returns (dare_delta, achieved_sparsity) where achieved_sparsity is the
    fraction of weights that were zeroed.

    Note: We use a deterministic seed per tensor for reproducibility. The seed
    is derived from the tensor's content hash so that re-running with the same
    weights produces identical masks.
    """
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    # Bernoulli mask: 1 = keep, 0 = drop
    mask = torch.bernoulli(
        torch.full(delta.shape, 1.0 - p, dtype=torch.float32),
        generator=generator,
    )

    dare_delta = delta * mask / (1.0 - p)

    # Compute achieved sparsity
    total = delta.numel()
    zeroed = (mask == 0).sum().item()
    achieved_sparsity = zeroed / total if total > 0 else 0.0

    return dare_delta, achieved_sparsity


def extract_head_delta(
    ft_model,
    base_model,
    layer: int,
    head: int,
    p: float,
    n_kv_heads: int,
) -> Tuple[torch.Tensor, float]:
    """
    Extract and DARE-sparsify the delta for a single attention head.

    For GQA models, W_O is shaped [n_heads, d_head, d_model] and W_V is
    shaped [n_kv_heads, d_model, d_head] (unexpanded KV weights).

    We extract:
      - W_O slice for head `head`: shape [d_head, d_model]
      - W_V slice for the corresponding KV group: shape [d_model, d_head]

    These are concatenated along dim=0 to form a single delta tensor for
    the head, then DARE is applied to the concatenated tensor.

    The KV group index for head h under GQA with repeat_kv = n_heads/n_kv_heads:
      kv_group = h // repeat_kv
    """
    repeat_kv = ft_model.cfg.n_heads // n_kv_heads

    # W_O: [n_heads, d_head, d_model]
    ft_W_O   = ft_model.blocks[layer].attn.W_O.detach().float()
    base_W_O = base_model.blocks[layer].attn.W_O.detach().float()

    # Slice head dimension
    ft_W_O_h   = ft_W_O[head]    # [d_head, d_model]
    base_W_O_h = base_W_O[head]  # [d_head, d_model]

    delta_W_O = compute_delta(ft_W_O_h, base_W_O_h)

    # W_V (unexpanded KV weights): [n_kv_heads, d_model, d_head]
    # Access via the private _W_V attribute for unexpanded weights
    ft_W_V_raw   = getattr(ft_model.blocks[layer].attn,   "_W_V", None)
    base_W_V_raw = getattr(base_model.blocks[layer].attn, "_W_V", None)

    if ft_W_V_raw is not None and base_W_V_raw is not None:
        kv_group = head // repeat_kv
        ft_W_V_h   = ft_W_V_raw[kv_group].detach().float()    # [d_model, d_head]
        base_W_V_h = base_W_V_raw[kv_group].detach().float()  # [d_model, d_head]
        delta_W_V  = compute_delta(ft_W_V_h, base_W_V_h)
        # Concatenate W_O and W_V deltas into a single tensor
        delta_combined = torch.cat([delta_W_O, delta_W_V.T], dim=0)
        # Shape: [d_head + d_head, d_model] = [2*d_head, d_model]
    else:
        # Fallback: W_V not available as unexpanded (standard MHA or older TL version)
        log.warning(
            "Layer %d head %d: _W_V not available, using W_O delta only",
            layer, head,
        )
        delta_combined = delta_W_O

    # Derive a deterministic seed from the layer/head identity
    seed = (layer * 1000 + head) % (2**31)
    dare_delta, sparsity = dare_sparsify(delta_combined, p=p, seed=seed)

    return dare_delta, sparsity


def extract_mlp_delta(
    ft_model,
    base_model,
    layer: int,
    p: float,
) -> Tuple[torch.Tensor, float]:
    """
    Extract and DARE-sparsify the delta for a full MLP layer.

    For gated MLP (SiLU gate, as in Llama/DeepSeek):
      W_in:   [d_model, d_mlp]
      W_out:  [d_mlp, d_model]
      W_gate: [d_model, d_mlp]

    All three matrices are concatenated into a single delta tensor, then
    DARE is applied uniformly.
    """
    mlp_ft   = ft_model.blocks[layer].mlp
    mlp_base = base_model.blocks[layer].mlp

    parts = []
    for attr in ("W_in", "W_out", "W_gate"):
        ft_w   = getattr(mlp_ft,   attr, None)
        base_w = getattr(mlp_base, attr, None)
        if ft_w is not None and base_w is not None:
            parts.append(compute_delta(ft_w.detach().float(), base_w.detach().float()))
        else:
            log.warning("Layer %d MLP: attribute %s not found, skipping", layer, attr)

    if not parts:
        raise ValueError(f"No MLP weight matrices found for layer {layer}")

    # Flatten and concatenate all MLP weight deltas
    delta_combined = torch.cat([p.flatten() for p in parts], dim=0)

    seed = (layer * 1000 + 999) % (2**31)
    dare_delta, sparsity = dare_sparsify(delta_combined, p=p, seed=seed)

    return dare_delta, sparsity


# ---------------------------------------------------------------------------
# Dry-run synthetic extraction
# ---------------------------------------------------------------------------

def run_dry_run_extraction(
    targets: List[TargetComponent],
    dare_p: float,
) -> Tuple[Dict[str, torch.Tensor], List[UORMetaRecord]]:
    """
    Generate synthetic delta tensors for dry-run mode.

    Produces realistic-shaped tensors with random values, applies DARE
    sparsification, and builds the full metadata records — exercising the
    complete pipeline logic without loading any model.
    """
    import random
    random.seed(42)
    torch.manual_seed(42)

    tensors: Dict[str, torch.Tensor] = {}
    meta_records: List[UORMetaRecord] = []

    for target in targets:
        tkey = target.tensor_key()

        if target.component_type == "attn_head":
            # Synthetic shape: [2*d_head, d_model] (W_O + W_V^T concatenated)
            shape = (2 * D_HEAD, D_MODEL)
        else:
            # Synthetic shape: flattened MLP delta
            # W_in [d_model, d_mlp] + W_out [d_mlp, d_model] + W_gate [d_model, d_mlp]
            shape = (D_MODEL * D_MLP + D_MLP * D_MODEL + D_MODEL * D_MLP,)

        # Synthetic delta: small random values (mimicking a fine-tuned delta)
        delta = torch.randn(shape) * 0.01

        seed = (target.layer * 1000 + (target.head or 999)) % (2**31)
        dare_delta, sparsity = dare_sparsify(delta, p=dare_p, seed=seed)

        tensors[tkey] = dare_delta

        # Content hash
        content_hash = hashlib.sha256(dare_delta.numpy().tobytes()).hexdigest()

        meta_records.append(UORMetaRecord(
            tensor_key=tkey,
            donor=MODEL_ID,
            vector_target="system1_system2_mode_switch",
            layer=target.layer,
            component_type=target.component_type,
            head=target.head,
            source_layers=[target.layer],
            uor_address=target.uor_address(),
            patch_effect_mean=target.patch_effect_mean,
            rank=target.rank,
            dare_p=dare_p,
            dare_rescale=round(1.0 / (1.0 - dare_p), 6),
            sparsity_achieved=round(sparsity, 6),
            tensor_shape=list(dare_delta.shape),
            tensor_dtype="float32",
            content_hash=content_hash,
            extraction_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        ))

        log.info(
            "  [dry-run] %s | shape=%s | sparsity=%.3f | uor=%s",
            tkey, list(dare_delta.shape), sparsity, target.uor_address(),
        )

    return tensors, meta_records


# ---------------------------------------------------------------------------
# Live extraction
# ---------------------------------------------------------------------------

def run_live_extraction(
    ft_model,
    base_model,
    targets: List[TargetComponent],
    dare_p: float,
) -> Tuple[Dict[str, torch.Tensor], List[UORMetaRecord]]:
    """
    Run DARE extraction on the live models for all target components.

    Returns (tensors_dict, meta_records) where tensors_dict maps tensor_key
    to the DARE-sparsified delta tensor, ready for safetensors serialisation.
    """
    n_kv_heads = ft_model.cfg.n_key_value_heads or ft_model.cfg.n_heads
    tensors: Dict[str, torch.Tensor] = {}
    meta_records: List[UORMetaRecord] = []

    for i, target in enumerate(targets):
        tkey = target.tensor_key()
        log.info(
            "[%d/%d] Extracting %s (rank=%d, layer=%d, effect=%.4f)",
            i + 1, len(targets), tkey, target.rank,
            target.layer, target.patch_effect_mean,
        )

        try:
            if target.component_type == "attn_head":
                dare_delta, sparsity = extract_head_delta(
                    ft_model, base_model,
                    layer=target.layer,
                    head=target.head,
                    p=dare_p,
                    n_kv_heads=n_kv_heads,
                )
            else:
                dare_delta, sparsity = extract_mlp_delta(
                    ft_model, base_model,
                    layer=target.layer,
                    p=dare_p,
                )
        except Exception as e:
            log.error("  FAILED: %s — %s", tkey, e)
            continue

        tensors[tkey] = dare_delta

        content_hash = hashlib.sha256(dare_delta.cpu().numpy().tobytes()).hexdigest()

        meta_records.append(UORMetaRecord(
            tensor_key=tkey,
            donor=MODEL_ID,
            vector_target="system1_system2_mode_switch",
            layer=target.layer,
            component_type=target.component_type,
            head=target.head,
            source_layers=[target.layer],
            uor_address=target.uor_address(),
            patch_effect_mean=target.patch_effect_mean,
            rank=target.rank,
            dare_p=dare_p,
            dare_rescale=round(1.0 / (1.0 - dare_p), 6),
            sparsity_achieved=round(sparsity, 6),
            tensor_shape=list(dare_delta.shape),
            tensor_dtype=str(dare_delta.dtype).replace("torch.", ""),
            content_hash=content_hash,
            extraction_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        ))

        log.info(
            "  -> shape=%s | sparsity=%.3f | uor=%s",
            list(dare_delta.shape), sparsity, target.uor_address(),
        )

    return tensors, meta_records


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_safetensors(
    tensors: Dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """Save the delta tensors to safetensors format."""
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # safetensors requires contiguous float32 or bfloat16 tensors
    tensors_to_save = {
        k: v.contiguous().to(torch.float32)
        for k, v in tensors.items()
    }

    save_file(tensors_to_save, str(output_path))
    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info("Saved safetensors: %s (%.2f MB)", output_path, size_mb)


def save_meta(
    manifest: ExtractionManifest,
    output_path: Path,
) -> None:
    """Save the UOR metadata sidecar to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
    log.info("Saved metadata sidecar: %s", output_path)


def build_manifest(
    meta_records: List[UORMetaRecord],
    tensors: Dict[str, torch.Tensor],
    dare_p: float,
    top_n: int,
    device: str,
    dtype: str,
    layer_head_map_status: str,
    runtime_seconds: float,
    is_dry_run: bool,
) -> ExtractionManifest:
    """Assemble the full ExtractionManifest from per-tensor records."""
    total_params = sum(t.numel() for t in tensors.values())
    total_nonzero = sum((t != 0).sum().item() for t in tensors.values())
    overall_sparsity = 1.0 - (total_nonzero / total_params) if total_params > 0 else 0.0

    manifest = ExtractionManifest(
        uor_donor=MODEL_ID,
        base_model=BASE_MODEL_ID,
        patching_mode="dry_run" if is_dry_run else "dare_targeted",
        device=device,
        dtype=dtype,
        dare_p=dare_p,
        dare_rescale=round(1.0 / (1.0 - dare_p), 6),
        top_n_components=top_n,
        signal_rank_threshold=top_n,
        num_tensors=len(tensors),
        total_parameters=total_params,
        total_nonzero_parameters=int(total_nonzero),
        overall_sparsity=round(overall_sparsity, 6),
        layer_head_map_path=str(LAYER_HEAD_MAP_PATH),
        layer_head_map_status=layer_head_map_status,
        safetensors_path=str(SAFETENSORS_OUTPUT),
        extraction_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        runtime_seconds=round(runtime_seconds, 2),
        tensors=[asdict(r) for r in meta_records],
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 2 — DARE extraction of mode-switching circuits "
                    "from DeepSeek-R1-Distill-8B with UOR metadata wrapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-path",
        default=MODEL_ID,
        help=f"Fine-tuned model HuggingFace id or local path (default: {MODEL_ID})",
    )
    p.add_argument(
        "--base-model-path",
        default=BASE_MODEL_ID,
        help=f"Base model HuggingFace id or local path (default: {BASE_MODEL_ID})",
    )
    p.add_argument(
        "--layer-head-map",
        default=str(LAYER_HEAD_MAP_PATH),
        help=f"Path to layer_head_map_deepseek.json (default: {LAYER_HEAD_MAP_PATH})",
    )
    p.add_argument(
        "--dare-p",
        type=float,
        default=DEFAULT_DARE_P,
        help=f"DARE drop probability (default: {DEFAULT_DARE_P})",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top-ranked components to extract (default: 20)",
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
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for safetensors and meta JSON (default: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate full pipeline logic without loading the model. "
            "Bypasses the LIVE status check on the layer_head_map and uses "
            "synthetic tensors."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir      = Path(args.output_dir)
    safetensors_out = output_dir / "deepseek_delta_uor.safetensors"
    meta_out        = output_dir / "deepseek_delta_uor_meta.json"
    lhm_path        = Path(args.layer_head_map)

    log.info("=" * 60)
    log.info("UOR-FORGE Stage 2 — Extract: DeepSeek-R1-Distill-8B")
    log.info("Target: System 1 / System 2 mode-switching circuits")
    log.info("DARE p=%.2f | top_n=%d | dry_run=%s",
             args.dare_p, args.top_n, args.dry_run)
    log.info("=" * 60)

    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load and validate Stage 1 artefact
    # ------------------------------------------------------------------
    if args.dry_run:
        # In dry-run, we still load the map but skip the LIVE status check
        # so the pipeline can be exercised without a live Stage 1 run.
        if lhm_path.exists():
            with open(lhm_path, encoding="utf-8") as f:
                layer_head_map = json.load(f)
            lhm_status = layer_head_map.get("_status", "UNKNOWN")
            log.info("[dry-run] Loaded layer_head_map (status=%s) — skipping LIVE check", lhm_status)
        else:
            log.warning("[dry-run] layer_head_map not found at %s — using synthetic targets", lhm_path)
            # Build a minimal synthetic map for dry-run
            layer_head_map = {
                "_status": "DRY_RUN_SYNTHETIC",
                "patching_mode": "dry_run",
                "all_components": [],
                "top_attention_heads": [
                    {"layer": l, "component_type": "attn_head", "head": h,
                     "patch_effect_mean": 0.5 - i*0.02, "rank": i+1}
                    for i, (l, h) in enumerate([
                        (12,15),(13,22),(10,7),(11,3),(14,8),
                        (15,1),(16,19),(9,11),(17,4),(10,21),
                    ])
                ],
                "top_mlp_layers": [
                    {"layer": l, "component_type": "mlp", "head": None,
                     "patch_effect_mean": 0.4 - i*0.02, "rank": 11+i}
                    for i, l in enumerate([12,13,11,14,10,15,16,9,17,8])
                ],
            }
            lhm_status = "DRY_RUN_SYNTHETIC"
    else:
        layer_head_map = load_and_validate_layer_head_map(lhm_path)
        lhm_status = layer_head_map.get("_status", "LIVE")

    # ------------------------------------------------------------------
    # Step 2: Select target components
    # ------------------------------------------------------------------
    targets = select_target_components(layer_head_map, top_n=args.top_n)

    if not targets:
        log.error("No target components found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Extract deltas
    # ------------------------------------------------------------------
    if args.dry_run:
        log.info("[dry-run] Running synthetic DARE extraction...")
        tensors, meta_records = run_dry_run_extraction(targets, dare_p=args.dare_p)
        device_used = "cpu"
        dtype_used  = "float32"
    else:
        ft_model, base_model = load_models(
            args.model_path, args.base_model_path,
            args.device, args.dtype,
        )
        log.info("Running live DARE extraction on %d target components...", len(targets))
        tensors, meta_records = run_live_extraction(
            ft_model, base_model, targets, dare_p=args.dare_p
        )
        device_used = args.device
        dtype_used  = args.dtype

    if not tensors:
        log.error("No tensors extracted. Exiting.")
        sys.exit(1)

    runtime = time.time() - t0

    # ------------------------------------------------------------------
    # Step 4: Build manifest and serialise
    # ------------------------------------------------------------------
    manifest = build_manifest(
        meta_records=meta_records,
        tensors=tensors,
        dare_p=args.dare_p,
        top_n=args.top_n,
        device=device_used,
        dtype=dtype_used,
        layer_head_map_status=lhm_status,
        runtime_seconds=runtime,
        is_dry_run=args.dry_run,
    )

    save_safetensors(tensors, safetensors_out)
    save_meta(manifest, meta_out)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Extraction complete in %.1fs", runtime)
    log.info("  Tensors extracted : %d", manifest.num_tensors)
    log.info("  Total parameters  : %d", manifest.total_parameters)
    log.info("  Non-zero params   : %d", manifest.total_nonzero_parameters)
    log.info("  Overall sparsity  : %.1f%%", manifest.overall_sparsity * 100)
    log.info("  Output            : %s", safetensors_out)
    log.info("  Metadata sidecar  : %s", meta_out)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
