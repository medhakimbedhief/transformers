import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import lazy_import_paged_flash_attention

# Global cache for flash_attn_with_kvcache function
_flash_attn_with_kvcache = None


def _get_flash_attn_with_kvcache(attn_implementation: str):
    """Lazily import flash_attn_with_kvcache from the kernel."""
    global _flash_attn_with_kvcache
    if _flash_attn_with_kvcache is None:
        from .hub_kernels import get_kernel

        # Extract actual kernel name (remove "paged|" prefix if present)
        kernel_name = attn_implementation.split("|")[-1]
        kernel = get_kernel(kernel_name)
        _flash_attn_with_kvcache = kernel.flash_attn_with_kvcache
    return _flash_attn_with_kvcache


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cache: PagedAttentionCache | None = None,
    cu_seq_lens_q=None,
    cu_seq_lens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    r"""Perform the forward pass of attention with paged key-value cache.

    This function handles the cache updates and performs the attention computation.
    For decode-only batches (when block_table is provided), uses flash_attn_with_kvcache
    for fused attention + cache update. Otherwise uses flash_attn_varlen_func.

    Args:
        q: (1, nheads, total_q, headdim), where total_q = total number of query tokens in the batch.
        k: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        v: (1, nheads_k, total_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seq_lens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seq_lens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        block_table: (num_groups, batch_size, max_blocks_per_seq), dtype int32. Block table for paged KV cache.
            If provided, uses flash_attn_with_kvcache for fused attention + cache update.
    """

    # Check if block_table is provided (decode-only batch)
    block_table = kwargs.pop("block_table", None)
    if block_table is not None and cache is not None:
        return _paged_attention_with_kvcache(
            module=module,
            q=q,
            k=k,
            v=v,
            cache=cache,
            cu_seq_lens_k=cu_seq_lens_k,
            max_seqlen_k=max_seqlen_k,
            block_table=block_table,
            **kwargs,
        )

    # Fallback: use flash_attn_varlen_func with read/write indices
    return _paged_attention_varlen(
        module,
        q,
        k,
        v,
        cache,
        cu_seq_lens_q,
        cu_seq_lens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    )


def _paged_attention_with_kvcache(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PagedAttentionCache,
    cu_seq_lens_k,
    max_seqlen_k,
    block_table: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Use flash_attn_with_kvcache for decode-only batches with paged KV cache.

    This path is faster for decode because:
    1. Fused attention + cache update in a single kernel
    2. No need to gather/scatter cache entries
    """
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"
    flash_attn_with_kvcache = _get_flash_attn_with_kvcache(module.config._attn_implementation)

    # Get layer group index for this layer
    group_idx, layer_idx_in_group = cache.layer_index_to_group_indices[module.layer_idx]

    # Get the paged KV cache for this layer, reshaped for the kernel
    # Cache shape: [num_pages, num_kv_heads, head_dim] -> [num_blocks, block_size, num_kv_heads, head_dim]
    k_cache = cache.key_cache[layer_idx_in_group].view(-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)
    v_cache = cache.value_cache[layer_idx_in_group].view(-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)

    # Reshape Q from [1, num_heads, batch_size, head_dim] to [batch_size, 1, num_heads, head_dim]
    # For decode-only, total_q == batch_size (1 token per request)
    batch_size = q.size(2)
    q_batched = q.squeeze(0).transpose(0, 1).unsqueeze(1).contiguous()  # [batch_size, 1, num_heads, head_dim]

    # Reshape K, V from [1, num_kv_heads, batch_size, head_dim] to [batch_size, 1, num_kv_heads, head_dim]
    k_new = k.squeeze(0).transpose(0, 1).unsqueeze(1).contiguous()  # [batch_size, 1, num_kv_heads, head_dim]
    v_new = v.squeeze(0).transpose(0, 1).unsqueeze(1).contiguous()  # [batch_size, 1, num_kv_heads, head_dim]

    # Compute cache_seqlens from cu_seq_lens_k (current cache length BEFORE adding new tokens)
    # cu_seq_lens_k is cumulative, so seqlens[i] = cu_seq_lens_k[i+1] - cu_seq_lens_k[i] - 1 (subtract 1 for the new token)
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]
    cache_seqlens = (cu_seq_lens_k[1:batch_size + 1] - cu_seq_lens_k[:batch_size] - 1).to(torch.int32)

    # Ensure block table is contiguous and contains valid indices
    # layer_block_table = layer_block_table.contiguous().to(torch.int32)

    # Call flash_attn_with_kvcache - this updates cache in-place and computes attention
    attn_output = flash_attn_with_kvcache(
        q_batched,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        page_table=block_table[group_idx],
        softmax_scale=module.scaling,
        causal=True,
        window_size=sliding_window,
    )

    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]

    # Reshape output from [batch_size, 1, num_heads, head_dim] to [batch_size, num_heads, head_dim]
    attn_output = attn_output.squeeze(1)

    return attn_output, None


def _paged_attention_varlen(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PagedAttentionCache | None,
    cu_seq_lens_q,
    cu_seq_lens_k,
    max_seqlen_q,
    max_seqlen_k,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Use flash_attn_varlen_func for prefill/mixed batches with read/write indices."""
    sliding_window = (-1, -1) if not getattr(module, "sliding_window", False) else (module.sliding_window - 1, 0)
    layer_type = "full_attention" if sliding_window == (-1, -1) else "sliding_attention"
    flash_attn_varlen_func = lazy_import_paged_flash_attention(module.config._attn_implementation)

    # .update changes the shape of k and v from [1, num_kv_heads, seqlen_kv, head_dim] to [-1, num_kv_heads, head_dim]
    if cache is not None:
        k, v = cache.update(
            key_states=k,
            value_states=v,
            layer_idx=module.layer_idx,
            read_index=kwargs["read_index"],
            write_index=kwargs["write_index"],
        )

    # Retrieve the cumulative sequence lengths for the current layer
    if isinstance(cu_seq_lens_k, dict):
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]

    custom_kwargs = {"s_aux": kwargs.get("s_aux")} if "s_aux" in kwargs else {}

    attn_output = flash_attn_varlen_func(
        q.transpose(1, 2).squeeze(0).contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seq_lens_q.to(torch.int32),
        cu_seq_lens_k.to(torch.int32).clone(),
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=module.scaling,
        causal=True,  # kind of a must, it automatically aligns the mask for q < k
        window_size=sliding_window,  # -1 means infinite context window
        **custom_kwargs,
    )
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]
    return attn_output, None
