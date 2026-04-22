import copy
import json
import time
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _suffix_metrics(logits: torch.Tensor):
    """Per-position metrics on suffix logits (B, V). Returns CPU-friendly 1D tensors."""
    probs = torch.softmax(logits, dim=-1)
    pmax, _ = probs.max(dim=-1)
    top2 = probs.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    return pmax, margin, ent


def _truncate_suffix_length(
    logits_suffix: torch.Tensor,
    policy: str,
    tau: float,
    risk_eps: float,
    score_beta4: float,
    block_size: int,
    num_confirmed: int,
) -> int:
    """Max k: first k positions in suffix pass policy (consecutive from start)."""
    if logits_suffix.shape[0] == 0:
        return 0
    pmax, margin, ent = _suffix_metrics(logits_suffix)
    L = logits_suffix.shape[0]
    if policy == "none" or policy == "full":
        return L
    if policy == "min_prob":
        for k in range(L):
            if float(pmax[k]) < tau:
                return k
        return L
    if policy == "risk":
        acc = 0.0
        for k in range(L):
            acc += float(1.0 - pmax[k])
            if acc > risk_eps + 1e-9:
                return k
        return L
    if policy == "score":
        rem = max(block_size - num_confirmed, 1)
        for k in range(L):
            j = float(k)
            r = (
                torch.log(pmax[k].clamp(min=1e-12))
                + margin[k]
                - ent[k]
                - score_beta4 * (j / rem)
            )
            if float(r) < tau:
                return k
        return L
    raise ValueError(f"unknown truncation policy: {policy}")


def _mean_int(xs: List[int]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / len(xs)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)

        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        # RoPE：attention 里 k = [k_ctx(ctx_len), k_q(q_len)]，cos/sin 必须与 k 的序列长度一致
        ctx_len = target_hidden.shape[1]
        q_len = hidden_states.shape[1]
        bsz = hidden_states.shape[0]
        if ctx_len == 0:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            draft0 = position_ids[:, 0:1]
            ctx_pos = draft0 - ctx_len + torch.arange(
                ctx_len,
                device=hidden_states.device,
                dtype=position_ids.dtype,
            ).view(1, -1)
            full_pos = torch.cat([ctx_pos, position_ids], dim=-1)
            rope_x = hidden_states.new_ones(
                bsz, ctx_len + q_len, hidden_states.size(-1)
            )
            position_embeddings = self.rotary_emb(rope_x, full_pos)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def get_last_decode_stats(self) -> dict:
        return getattr(self, "_last_decode_stats", {})

    def _emit_draft_topk(
        self,
        logits: torch.Tensor,
        topk_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将 (B, seq, V) 上每个位置的 top-k 概率与 decode 文本写入文件（需已设置 _draft_topk_file）。

        每条记录为一块多行 JSON：`topk` 内每个 token 单独一行（紧凑对象），便于阅读。
        """
        fp = getattr(self, "_draft_topk_file", None)
        tok = getattr(self, "_draft_topk_tokenizer", None)
        if fp is None or tok is None:
            return
        k = max(1, int(getattr(self, "_draft_topk_k", 2)))
        self._draft_topk_seq = int(getattr(self, "_draft_topk_seq", 0)) + 1
        row_meta: Dict[str, Any] = {
            "kind": "draft_logits_topk",
            "call_id": self._draft_topk_seq,
            "verify_step": int(getattr(self, "_draft_topk_verify_step", -1)),
            "block_start_abs": int(getattr(self, "_draft_topk_block_start", -1)),
        }
        if topk_meta:
            row_meta.update(topk_meta)
        probs = torch.softmax(logits[0].float(), dim=-1)
        seq_len = probs.shape[0]
        vals, idx = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
        positions_out: List[dict] = []
        for i in range(seq_len):
            if i == 0:
                continue
            top_list = []
            for j in range(vals.shape[1]):
                tid = int(idx[i, j].item())
                p = float(vals[i, j].item())
                text = tok.decode([tid], skip_special_tokens=False)
                top_list.append({"token_id": tid, "prob": round(p, 8), "decoded": text})
            positions_out.append({"position_in_block": i, "topk": top_list})

        meta_json = json.dumps(row_meta, ensure_ascii=False, indent=2)
        parts: List[str] = ["{\n  \"meta\": ", meta_json.replace("\n", "\n  "), ",\n  \"positions\": ["]
        for pi, pos in enumerate(positions_out):
            parts.append("\n    {\n      \"position_in_block\": ")
            parts.append(str(pos["position_in_block"]))
            parts.append(",\n      \"topk\": [")
            for ti, t in enumerate(pos["topk"]):
                parts.append("\n        ")
                parts.append(json.dumps(t, ensure_ascii=False))
                if ti < len(pos["topk"]) - 1:
                    parts.append(",")
            parts.append("\n      ]\n    }")
            if pi < len(positions_out) - 1:
                parts.append(",")
        parts.append("\n  ]\n}\n")
        fp.write("".join(parts))
        fp.flush()

    def _emit_draft_profile_target_accept(
        self,
        accept_len_tokens: int,
        commit_len_pre_verify: int,
        accepted_token_ids: List[int],
        correction_token_id: int,
    ) -> None:
        """本轮 target 验证结束：接收长度、与 draft 对齐的连续接受 token、首处分歧处 target 采样 token。"""
        fp = getattr(self, "_draft_topk_file", None)
        if fp is None:
            return
        tok = getattr(self, "_draft_topk_tokenizer", None)
        meta: Dict[str, Any] = {
            "kind": "target_accept",
            "verify_step": int(getattr(self, "_draft_topk_verify_step", -1)),
            "block_start_abs": int(getattr(self, "_draft_topk_block_start", -1)),
            "accept_len_tokens": int(accept_len_tokens),
            "commit_len_pre_verify": int(commit_len_pre_verify),
            "accepted_token_ids": accepted_token_ids,
            "correction_token_id": int(correction_token_id),
        }
        if tok is not None:
            meta["accepted_decoded_concat"] = tok.decode(
                accepted_token_ids, skip_special_tokens=False
            )
            meta["verified_tokens"] = [
                {
                    "token_id": int(tid),
                    "decoded": tok.decode([tid], skip_special_tokens=False),
                }
                for tid in accepted_token_ids
            ]
            meta["correction_decoded"] = tok.decode(
                [correction_token_id], skip_special_tokens=False
            )
        fp.write(json.dumps({"meta": meta}, ensure_ascii=False, indent=2) + "\n\n")
        fp.flush()

    def _draft_forward_block_logits(
        self,
        target: nn.Module,
        block_output_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values_draft: DynamicCache,
        start: int,
        block_size: int,
        topk_meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        noise_embedding = target.model.embed_tokens(block_output_ids)
        # 必须与 noise_embedding 长度一致；勿用 get_seq_length 切片（缓存与 start 可能短暂不一致）
        q_blk = noise_embedding.shape[1]
        pos_block = position_ids[:, start : start + q_blk]
        draft_hidden = self(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=pos_block,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        logits = target.lm_head(draft_hidden)
        if getattr(self, "_draft_topk_file", None) is not None:
            self._emit_draft_topk(logits, topk_meta)
        past_key_values_draft.crop(start)
        return logits

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        *,
        max_block_inner_iters: int = 1,
        truncation_policy: str = "full",
        trunc_tau: float = 0.85,
        trunc_risk_eps: float = 0.35,
        trunc_score_beta4: float = 0.15,
        inner_fallback: str = "one",
        use_draft_tree: bool = False,
        draft_tree_branches: int = 3,
        debug_dir: Optional[str] = None,
        tokenizer=None,
        draft_topk_file=None,
        draft_topk: int = 2,
        calibration_records: Optional[List[Dict[str, Any]]] = None,
        record_posthoc_suffix_refine: bool = False,
    ):
        """Speculative generation with optional DFlash++ in-block iteration and draft tree.

        When ``record_posthoc_suffix_refine`` is True (no draft tree), after each target verify
        we measure a counterfactual: freeze the accepted prefix, remask the suffix, run one
        draft forward to refill the tail, target-verify again, and record accept-length gain.
        Steps with accept length 1 (only block index 0) are skipped: same draft input as the
        first refine round, so no extra information. KV caches are snapshot/restored so the
        real decode path is unchanged.
        """
        self.eval()
        max_block_inner_iters = max(1, min(3, int(max_block_inner_iters)))
        t_target = 0.0
        t_draft = 0.0
        steps = 0
        acceptance_lengths: List[int] = []
        inner_kt_rounds: List[List[int]] = []
        inner_iters_per_block: List[int] = []
        commit_lens_pre_verify: List[int] = []
        accept_lens_actual: List[int] = []
        trunc_calib_hits: List[int] = []
        second_plus_kt_sum: List[int] = []
        tree_branch_picks: List[int] = []
        posthoc_suffix_refine_gains: List[Optional[int]] = []
        posthoc_suffix_refine_pairs: List[List[int]] = []
        posthoc_suffix_skipped_anchor_only = 0
        t_target_posthoc_extra = 0.0

        device = target.device
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size
        mask_id = self.mask_token_id

        self._draft_topk_seq = 0
        self._calibration_pending: Optional[Dict[str, Any]] = None
        if draft_topk_file is not None and tokenizer is not None:
            self._draft_topk_file = draft_topk_file
            self._draft_topk_tokenizer = tokenizer
            self._draft_topk_k = int(draft_topk)
        else:
            self._draft_topk_file = None
            self._draft_topk_tokenizer = None
            self._draft_topk_k = 2

        output_ids = torch.full(
            (1, max_length + block_size),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=device, dtype=torch.long
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        t0 = time.perf_counter()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_target += time.perf_counter() - t0

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        start = input_ids.shape[1]

        def refine_block_linear(
            block_tensor: torch.Tensor,
            policy: str,
            init_confirmed: int = 1,
        ) -> tuple[torch.Tensor, List[int], int]:
            """In-place refine block_tensor[0]; returns (block, k_list, num_confirmed)."""
            nonlocal t_draft
            k_list: List[int] = []
            num_confirmed = int(init_confirmed)
            inner_used = 0
            while (
                inner_used < max_block_inner_iters
                and num_confirmed < block_size
            ):
                block_tensor[:, num_confirmed:] = mask_id
                t_d = time.perf_counter()
                full_lm = self._draft_forward_block_logits(
                    target,
                    block_tensor,
                    target_hidden,
                    position_ids,
                    past_key_values_draft,
                    start,
                    block_size,
                    topk_meta={
                        "stage": "refine_inner",
                        "inner_used": inner_used,
                        "num_confirmed_before": num_confirmed,
                    },
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_draft += time.perf_counter() - t_d

                logits_suf = full_lm[0, num_confirmed:block_size, :]
                if logits_suf.shape[0] == 0:
                    break
                if (
                    calibration_records is not None
                    and inner_used == 0
                    and init_confirmed == 1
                    and not use_draft_tree
                ):
                    pm, mg, et = _suffix_metrics(logits_suf)
                    probs_suf = torch.softmax(logits_suf.float(), dim=-1)
                    t2v, t2i = torch.topk(probs_suf, min(2, probs_suf.shape[-1]), dim=-1)
                    self._calibration_pending = {
                        "pmax": pm.detach().cpu().tolist(),
                        "margin": mg.detach().cpu().tolist(),
                        "ent": et.detach().cpu().tolist(),
                        "top2_prob": t2v.detach().cpu().tolist(),
                        "top2_token_id": t2i.detach().cpu().tolist(),
                        "num_confirmed_start": int(num_confirmed),
                    }
                pred_suf = torch.argmax(logits_suf, dim=-1)
                trunc_pol = "full" if policy == "full" else policy
                k = _truncate_suffix_length(
                    logits_suf,
                    trunc_pol,
                    trunc_tau,
                    trunc_risk_eps,
                    trunc_score_beta4,
                    block_size,
                    num_confirmed,
                )
                if k == 0 and inner_fallback == "one" and logits_suf.shape[0] > 0:
                    k = 1
                if k == 0:
                    break
                block_tensor[
                    :, num_confirmed : num_confirmed + k
                ] = pred_suf[:k].unsqueeze(0)
                k_list.append(int(k))
                num_confirmed += k
                inner_used += 1
                if policy == "full" or max_block_inner_iters == 1:
                    break
            if num_confirmed < block_size:
                block_tensor[:, num_confirmed:] = mask_id
                t_d = time.perf_counter()
                full_lm = self._draft_forward_block_logits(
                    target,
                    block_tensor,
                    target_hidden,
                    position_ids,
                    past_key_values_draft,
                    start,
                    block_size,
                    topk_meta={
                        "stage": "refine_fill_tail",
                        "num_confirmed_before": num_confirmed,
                    },
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_draft += time.perf_counter() - t_d
                logits_suf = full_lm[0, num_confirmed:block_size, :]
                if logits_suf.shape[0] > 0:
                    pred_suf = torch.argmax(logits_suf, dim=-1)
                    block_tensor[:, num_confirmed:block_size] = pred_suf.unsqueeze(0)
                    num_confirmed = block_size
            return block_tensor, k_list, num_confirmed

        def draft_tree_block(
            block_init: torch.Tensor,
            policy: str,
        ) -> tuple[torch.Tensor, List[int], int, int]:
            """Probe + r±1 cuts; return best block by mean log pmax, branch index."""
            nonlocal t_draft
            t_d = time.perf_counter()
            probe = block_init.clone()
            probe[:, 1:] = mask_id
            full_lm = self._draft_forward_block_logits(
                target,
                probe,
                target_hidden,
                position_ids,
                past_key_values_draft,
                start,
                block_size,
                topk_meta={"stage": "draft_tree_probe"},
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_draft += time.perf_counter() - t_d
            logits_all = full_lm[0, 1:block_size, :]
            pred_all = torch.argmax(logits_all, dim=-1)
            pmax_all, _, _ = _suffix_metrics(logits_all)
            r = int(torch.argmin(pmax_all).item()) + 1
            cuts = {r - 1, r, r + 1}
            cuts = {c for c in cuts if 1 <= c <= block_size - 1}
            # 原逻辑在 max(cuts)==block_size-1 时 add 的是已存在元素，len 不变 → 死循环（CPU 空转，GPU 0%）
            while len(cuts) < draft_tree_branches and len(cuts) < block_size - 1:
                n_before = len(cuts)
                hi = max(cuts)
                if hi < block_size - 1:
                    cuts.add(hi + 1)
                if len(cuts) == n_before:
                    lo = min(cuts)
                    if lo > 1:
                        cuts.add(lo - 1)
                if len(cuts) == n_before:
                    break
            ordered = sorted(cuts)[: max(1, draft_tree_branches)]

            best_score = -1e9
            best_block = block_init.clone()
            best_klist: List[int] = []
            best_commit = 1
            best_bi = 0
            for bi, tcut in enumerate(ordered):
                cand = block_init.clone()
                cand[:, 1 : tcut + 1] = pred_all[:tcut].unsqueeze(0)
                cand[:, tcut + 1 :] = mask_id
                refined, klist, ncom = refine_block_linear(
                    cand.clone(), policy, init_confirmed=tcut + 1
                )
                t_d2 = time.perf_counter()
                lm2 = self._draft_forward_block_logits(
                    target,
                    refined,
                    target_hidden,
                    position_ids,
                    past_key_values_draft,
                    start,
                    block_size,
                    topk_meta={
                        "stage": "draft_tree_score",
                        "branch": bi,
                        "tcut": int(tcut),
                    },
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_draft += time.perf_counter() - t_d2
                p2, _, _ = _suffix_metrics(lm2[0, 1:block_size, :])
                score = float(torch.log(p2.clamp(min=1e-12)).mean().item())
                if score > best_score:
                    best_score = score
                    best_block = refined
                    best_klist = klist
                    best_commit = ncom
                    best_bi = bi
            return best_block, best_klist, best_commit, best_bi

        while start < max_length:
            self._draft_topk_verify_step = int(steps)
            self._draft_topk_block_start = int(start)
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            pol = truncation_policy
            if max_block_inner_iters <= 1 and not use_draft_tree:
                pol = "full"

            if use_draft_tree:
                b, k_list, commit_len, bidx = draft_tree_block(
                    block_output_ids, pol
                )
                block_output_ids = b
                tree_branch_picks.append(bidx)
            else:
                block_output_ids, k_list, commit_len = refine_block_linear(
                    block_output_ids, pol
                )

            inner_kt_rounds.append(k_list)
            inner_iters_per_block.append(len(k_list))
            if len(k_list) >= 2:
                second_plus_kt_sum.append(int(sum(k_list[1:])))
            commit_lens_pre_verify.append(int(commit_len))

            snap_t_before: Optional[DynamicCache] = None
            snap_t_after_first: Optional[DynamicCache] = None
            snap_d_after_refine: Optional[DynamicCache] = None
            if record_posthoc_suffix_refine and not use_draft_tree:
                snap_t_before = copy.deepcopy(past_key_values_target)
                snap_d_after_refine = copy.deepcopy(past_key_values_draft)

            t0 = time.perf_counter()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_target += time.perf_counter() - t0

            if record_posthoc_suffix_refine and not use_draft_tree:
                snap_t_after_first = copy.deepcopy(past_key_values_target)

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            accept_lens_actual.append(int(acceptance_length + 1))

            posthoc_gain_block: Optional[int] = None
            _acc1 = int(acceptance_length + 1)
            if (
                record_posthoc_suffix_refine
                and not use_draft_tree
                and _acc1 < block_size
                and _acc1 <= 1
            ):
                posthoc_suffix_skipped_anchor_only += 1
            # acc1==1 时仅块内位置 0 被接受，与首轮 refine 的「仅确认块首、后缀全 mask」相同；
            # 同条件 target_hidden/KV 下二次 suffix 草稿与首轮等价（greedy 下无新信息），跳过。
            if (
                record_posthoc_suffix_refine
                and not use_draft_tree
                and snap_t_before is not None
                and snap_t_after_first is not None
                and snap_d_after_refine is not None
                and _acc1 < block_size
                and _acc1 > 1
            ):
                past_key_values_target = copy.deepcopy(snap_t_before)
                past_key_values_draft = copy.deepcopy(snap_d_after_refine)
                b2 = block_output_ids.clone()
                b2[:, _acc1:] = mask_id
                t_d = time.perf_counter()
                full_lm_ph = self._draft_forward_block_logits(
                    target,
                    b2,
                    target_hidden,
                    position_ids,
                    past_key_values_draft,
                    start,
                    block_size,
                    topk_meta={
                        "stage": "posthoc_suffix_refine",
                        "frozen_prefix_len": _acc1,
                    },
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_draft += time.perf_counter() - t_d
                logits_tail = full_lm_ph[0, _acc1:block_size, :]
                if logits_tail.shape[0] > 0:
                    pred_tail = torch.argmax(logits_tail, dim=-1)
                    b2[:, _acc1:block_size] = pred_tail.unsqueeze(0)
                t0_ph = time.perf_counter()
                output_ph = target(
                    b2,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt_ph = time.perf_counter() - t0_ph
                t_target_posthoc_extra += dt_ph
                posterior_ph = sample(output_ph.logits, temperature)
                al2 = (
                    (b2[:, 1:] == posterior_ph[:, :-1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
                acc2 = int(al2 + 1)
                posthoc_gain_block = int(acc2 - _acc1)
                posthoc_suffix_refine_pairs.append([_acc1, posthoc_gain_block])
                past_key_values_target = copy.deepcopy(snap_t_after_first)
                past_key_values_draft = copy.deepcopy(snap_d_after_refine)
            posthoc_suffix_refine_gains.append(posthoc_gain_block)
            trunc_calib_hits.append(
                1 if commit_len <= acceptance_length + 1 else 0
            )
            _n_acc = int(acceptance_length + 1)
            _acc_ids = block_output_ids[0, :_n_acc].detach().cpu().tolist()
            _corr = int(posterior[0, acceptance_length].item())
            self._emit_draft_profile_target_accept(
                _n_acc, int(commit_len), _acc_ids, _corr
            )
            if calibration_records is not None and self._calibration_pending is not None:
                row = dict(self._calibration_pending)
                row["accept_len"] = _n_acc
                row["verify_step"] = int(steps)
                row["block_start_abs"] = int(start)
                calibration_records.append(row)
                self._calibration_pending = None

            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            steps += 1

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_id]
        if stop_token_ids is not None:
            st = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], st
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        mean_k2p = _mean_int(second_plus_kt_sum)
        calib_rate = (
            float(sum(trunc_calib_hits)) / len(trunc_calib_hits)
            if trunc_calib_hits
            else 0.0
        )
        gains_only = [g for g in posthoc_suffix_refine_gains if g is not None]
        gains_from_pairs = [int(p[1]) for p in posthoc_suffix_refine_pairs]
        mean_posthoc_gain_corrected = (
            _mean_int(gains_from_pairs) if gains_from_pairs else 0.0
        )

        self._last_decode_stats = {
            "accept_lengths": acceptance_lengths,
            "target_total_time": t_target,
            "draft_total_time": t_draft,
            "steps": steps,
            "inner_kt_rounds": inner_kt_rounds,
            "inner_iters_per_block": inner_iters_per_block,
            "mean_inner_iters": _mean_int(inner_iters_per_block),
            "second_plus_kt_values": second_plus_kt_sum,
            "mean_second_plus_kt": mean_k2p,
            "commit_lens_pre_verify": commit_lens_pre_verify,
            "accept_lens_actual": accept_lens_actual,
            "trunc_commit_le_accept_rate": calib_rate,
            "tree_branch_picks": tree_branch_picks,
            "max_block_inner_iters": max_block_inner_iters,
            "truncation_policy": truncation_policy,
            "trunc_tau": trunc_tau,
            "use_draft_tree": use_draft_tree,
            "record_posthoc_suffix_refine": record_posthoc_suffix_refine,
            "posthoc_suffix_refine_gains": posthoc_suffix_refine_gains,
            "posthoc_suffix_refine_pairs": posthoc_suffix_refine_pairs,
            "mean_posthoc_suffix_accept_gain": mean_posthoc_gain_corrected,
            "mean_posthoc_suffix_accept_gain_corrected": mean_posthoc_gain_corrected,
            "n_posthoc_suffix_refine_executed": len(posthoc_suffix_refine_pairs),
            "n_posthoc_suffix_skipped_anchor_only": posthoc_suffix_skipped_anchor_only,
            "n_posthoc_suffix_events": len(gains_only),
            "target_posthoc_extra_time": t_target_posthoc_extra,
        }
        self._draft_topk_file = None
        self._draft_topk_tokenizer = None
        self._calibration_pending = None
        return output_ids
