# coding=utf-8
"""DFlash++ training wrapper: DFlash block CE + iterative completion (L_con)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.dflash import create_dflash_block_mask
from specforge.modeling.draft.dflash import DFlashDraftModel


class OnlineDFlashPPModel(nn.Module):
    """DFlash++ online training: L_dflash (original) + λ · L_con (clean-prefix completion).

    One draft forward stacks both masks on the batch dimension (2×batch), then splits logits.
    L_con 前缀长度 p ∈ {K,…,B-1}，K=`lcon_min_prefix_len`（块内前 K 个位置恒为干净，与推理一致);
    P(p) ∝ exp(-w·(p-1-b)²) 仅在上述集合上归一化 (w=0 → 均匀).
    """

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dflash_loss_weight: float = 1.0,
        completion_loss_weight: float = 1.0,
        completion_gamma: Optional[float] = None,
        completion_prefix_sample_weight: float = 1.0,
        completion_prefix_sample_bias: float = 0.0,
        lcon_min_prefix_len: int = 3,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.dflash_loss_weight = dflash_loss_weight
        self.completion_loss_weight = completion_loss_weight
        self.completion_gamma = completion_gamma
        self.completion_prefix_sample_weight = completion_prefix_sample_weight
        self.completion_prefix_sample_bias = completion_prefix_sample_bias
        self.lcon_min_prefix_len = int(lcon_min_prefix_len)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed_dflash(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Original DFlash: only block position 0 is non-mask at anchor."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _create_noise_embed_completion(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Clean prefix of length p per block; positions [p, B) are mask (design §5.2)."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )
        noise_view = noise_ids.view(bsz, n, bs)

        offsets = torch.arange(bs, device=device).view(1, 1, -1)
        seq_idx = anchor_positions.unsqueeze(-1) + offsets
        seq_idx = seq_idx.clamp(0, seq_len - 1)
        gathered = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1), 2, seq_idx
        )

        p = prefix_lengths.unsqueeze(-1).clamp(0, bs)
        is_clean = (offsets < p) & block_keep_mask.unsqueeze(-1)
        noise_view[:] = torch.where(
            is_clean,
            gathered,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _sample_prefix_lengths(
        self, bsz: int, n: int, device: torch.device
    ) -> torch.Tensor:
        """Sample p ∈ {K,…,B-1}, K=`lcon_min_prefix_len`. logits_p = -w * (p - 1 - b)^2.

        块内前 K 个位置不作为「mask 起点」被采样。Note: linear -w*(p-1-b) is softmax-invariant
        in b. Squared form makes b shift the mode: peak near p ≈ 1 + b (clipped to allowed set).
        w=0 → uniform over allowed p; larger w → sharper around that mode.
        """
        bs = self.block_size
        min_p = int(self.lcon_min_prefix_len)
        w = self.completion_prefix_sample_weight
        b = self.completion_prefix_sample_bias
        idx = torch.arange(min_p, bs, device=device, dtype=torch.float32)
        centered = idx - 1.0 - b
        logits = -w * centered * centered
        probs = F.softmax(logits, dim=-1)
        total = bsz * n
        flat = torch.multinomial(probs, num_samples=total, replacement=True)
        return (flat + min_p).view(bsz, n).long()

    def _loss_dflash_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(
                -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy

    def _loss_completion_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """L_con: CE on positions j >= p with normalized exp weights (design §5.3, §6)."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        bs = self.block_size
        n = anchor_positions.shape[1]

        label_offsets = torch.arange(0, bs, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1),
            2,
            safe_label_indices,
        )

        pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
        p_expand = prefix_lengths.unsqueeze(-1).clamp(0, bs)
        train_pos = (pos_in_block >= p_expand) & block_keep_mask.unsqueeze(-1)
        train_pos = train_pos & valid_label_mask

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n, -1),
            2,
            safe_label_indices,
        )
        train_pos = train_pos & (original_loss_mask_gathered > 0.5)
        has_supervised = train_pos.any(dim=-1)

        d = (pos_in_block - p_expand + 1).float().clamp(min=1.0)
        if self.completion_gamma is not None and self.completion_gamma > 0:
            w = torch.exp(-(d - 1.0) / self.completion_gamma)
        else:
            w = torch.ones_like(d)

        w = w * train_pos.float()
        sum_w = w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        tilde_w = w / sum_w

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction="none",
        ).view(bsz, n, bs)

        block_loss = (ce * tilde_w).sum(dim=-1)
        valid_block = block_keep_mask & has_supervised
        denom_blocks = valid_block.float().sum().clamp(min=1.0)
        loss = (block_loss * valid_block.float()).sum() / denom_blocks

        with torch.no_grad():
            pred_ids = torch.argmax(logits, dim=-1).view(bsz, n, bs)
            correct = (pred_ids == target_ids) & train_pos
            accuracy = correct.float().sum() / train_pos.float().sum().clamp(min=1.0)

        return loss, accuracy

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_loss, accuracy_for_logging, loss_dflash, loss_con).

        Single draft forward: batch-dim concat of DFlash vs completion noise; logits split.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        k = int(self.lcon_min_prefix_len)
        if k < 1 or k >= self.block_size:
            raise ValueError(
                f"lcon_min_prefix_len (K) must satisfy 1 <= K < block_size; got K={k}, B={self.block_size}."
            )

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        n = anchor_positions.shape[1]

        prefix_lengths = self._sample_prefix_lengths(bsz, n, device)

        noise_df = self._create_noise_embed_dflash(
            input_ids, anchor_positions, block_keep_mask
        )
        noise_con = self._create_noise_embed_completion(
            input_ids, anchor_positions, block_keep_mask, prefix_lengths
        )
        noise_cat = torch.cat([noise_df, noise_con], dim=0)

        hidden_cat = torch.cat([hidden_states, hidden_states], dim=0)
        anchor_cat = torch.cat([anchor_positions, anchor_positions], dim=0)
        block_keep_cat = torch.cat([block_keep_mask, block_keep_mask], dim=0)

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        full_cat = torch.cat([full_position_ids, full_position_ids], dim=0)

        dflash_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_cat,
            block_keep_mask=block_keep_cat,
            S=seq_len,
            block_size=self.block_size,
            device=device,
        )

        output_hidden = self.draft_model(
            position_ids=full_cat,
            noise_embedding=noise_cat,
            target_hidden=hidden_cat,
            attention_mask=dflash_attn_mask,
        )
        logits = self.lm_head(output_hidden)
        logits_df = logits[:bsz]
        logits_con = logits[bsz:]

        loss_df, acc_df = self._loss_dflash_from_logits(
            logits_df,
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
        )
        loss_con, acc_con = self._loss_completion_from_logits(
            logits_con,
            input_ids,
            loss_mask,
            anchor_positions,
            block_keep_mask,
            prefix_lengths,
        )

        total = (
            self.dflash_loss_weight * loss_df
            + self.completion_loss_weight * loss_con
        )
        acc_mean = 0.5 * (acc_df + acc_con)

        return total, acc_mean, loss_df, loss_con
