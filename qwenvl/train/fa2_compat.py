"""Compatibility helpers for Qwen3.5 and Transformers FlashAttention2."""

from __future__ import annotations

from typing import Callable


QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})


def patch_qwen35_flash_attention_position_ids() -> bool:
    """Keep Qwen3.5 MRoPE position IDs out of FA2 packed-sequence detection.

    Qwen3.5 passes ``position_ids`` with shape ``[3, batch, seq_len]``. Some
    Transformers releases assume a 2-D tensor in ``_is_packed_sequence`` and
    can mistake these MRoPE positions for a flattened packed sequence. That
    produces invalid cumulative sequence lengths in the FA2 varlen path.

    A real packed sequence uses 2-D position IDs. Rejecting tensors with more
    dimensions preserves the dense no-padding FA2 path. The patch is local to
    the current Python process and is safe to call more than once.
    """
    from transformers import modeling_flash_attention_utils as fa_utils

    original: Callable | None = getattr(fa_utils, "_is_packed_sequence", None)
    if original is None or getattr(original, "_RSI_NAV_qwen35_mrope_safe", False):
        return False

    def mrope_safe_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and getattr(position_ids, "ndim", 0) > 2:
            return False
        return original(position_ids, batch_size)

    mrope_safe_is_packed_sequence._RSI_NAV_qwen35_mrope_safe = True
    fa_utils._is_packed_sequence = mrope_safe_is_packed_sequence
    return True


def maybe_patch_qwen35_flash_attention(
    model_type: str,
    attn_implementation: str,
) -> bool:
    """Install the MRoPE fix when loading Qwen3.5 with FlashAttention2."""
    if model_type not in QWEN35_MODEL_TYPES:
        return False
    if attn_implementation != "flash_attention_2":
        return False
    return patch_qwen35_flash_attention_position_ids()
