# coding=utf-8
"""DFlash 草稿注意力中的 RoPE：与训练/推理的 position 布局解耦，单独维护。"""

import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q、K 应用旋转位置编码。

    期望 ``cos`/`sin`` 的序列维与 K 的序列长度 ``k.size(-2)`` 一致（K = [k_ctx, k_draft]）。
    Q 仅覆盖 draft 段，与 ``cos`/`sin`` 的**最后** ``q.size(-2)`` 个位置对齐。

    若因上游 bug 导致 cos 比 k 长，会取最后 ``k_len`` 维以与 k 对齐；更短时抛错。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    k_len = k.size(-2)
    cos_k = cos.size(-2)

    if cos_k > k_len:
        cos = cos[..., -k_len:, :]
        sin = sin[..., -k_len:, :]
    elif cos_k < k_len:
        raise ValueError(
            f"apply_rotary_pos_emb: cos/sin seq {cos_k} < k seq {k_len}. "
            "Check DFlash forward: position_ids length must match ctx_len + q_len."
        )

    if q_len > k_len:
        raise ValueError(
            f"apply_rotary_pos_emb: q_len {q_len} > k_len {k_len} (unexpected for DFlash)."
        )

    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
