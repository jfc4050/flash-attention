import math
import torch
from apex.transformer.functional import FusedScaleMaskSoftmax
from apex.transformer import AttnMaskType


def attention_mask_func(attention_scores, attention_mask):
    attention_mask = attention_mask.bool()
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


scale_mask_softmax = FusedScaleMaskSoftmax(
    False, True, # Only takes effect for the unfused fallback kernel
    AttnMaskType.causal,
    True,
    attention_mask_func,
    True,
    None
)

def attn_fused_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
):
    assert causal, "only a parameter for API compatibility"

    # attention_scores
    batch_size, partitions, query_length, head_dim = query.shape
    key_length = key.size(2)
    output_size = (batch_size, partitions, query_length, key_length)

    attn_scale = 1 / math.sqrt(head_dim)

    query = query.reshape((batch_size * partitions, query_length, head_dim))
    key = key.reshape((batch_size * partitions, key_length, head_dim)).transpose(-1, -2)
    attn_weights = torch.empty(
        batch_size * partitions,
        query_length,
        key_length,
        dtype=query.dtype,
        device=query.device)
    attn_weights.baddbmm_(query, key, beta=0.0, alpha=attn_scale)
    attn_weights = attn_weights.view(*output_size)

    if bias is not None:
        attn_weights += bias.expand(*output_size)

    attn_weights = scale_mask_softmax(attn_weights, None)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    attn_output = torch.matmul(attn_weights, value)

    return attn_output

