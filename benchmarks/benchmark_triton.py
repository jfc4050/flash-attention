import math
import torch
from torch.utils.benchmark import Compare

from flash_attn.utils.benchmark import benchmark_all
from flash_attn.flash_attn_triton import flash_attn_func

WARMUP_REPS = 30


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen, nheads, head_dim)
        k: (batch_size, seqlen, nheads, head_dim)
        v: (batch_size, seqlen, nheads, head_dim)
        attn_mask: (batch_size, nheads, seqlen, seqlen)
        dropout_p: float
        causal: bool
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, nheads, head_dim = q.shape

    scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(head_dim))
    if bias is not None:
        scores = scores + bias
    if causal:
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device), 1
        )
        scores.masked_fill_(causal_mask, float("-inf"))

    attention = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attention = torch.nn.functional.dropout(attention, dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output


def run_benchmark(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    dropout_p: float,
    dtype: torch.dtype,
) -> list:
    device = "cuda"

    # setup inputs
    q = torch.randn(
        (batch_size, seq_len, n_heads, head_dim),
        requires_grad=True,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (batch_size, seq_len, n_heads, head_dim),
        requires_grad=True,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        (batch_size, seq_len, n_heads, head_dim),
        requires_grad=True,
        device=device,
        dtype=dtype,
    )
    bias = torch.randn(
        (batch_size, n_heads, seq_len, seq_len),
        requires_grad=False,
        device=device,
        dtype=dtype,
    )

    # run benchmarks
    sub_label = f"({batch_size}, {n_heads}, {seq_len}, {head_dim}), p={dropout_p}, causal={causal}, dtype={dtype}"
    triton_fn = lambda q, k, v, bias, causal, dropout_p: flash_attn_func(
        q, k, v, bias, causal, dropout_p
    )
    ref_fn = lambda q, k, v, bias, causal, dropout_p: attention_ref(
        q, k, v, bias, causal, dropout_p
    )
    triton_benchmark_results = benchmark_all(
        triton_fn,
        q,
        k,
        v,
        bias,
        causal,
        dropout_p,
        repeats=WARMUP_REPS,
        desc="FlashAttention",
        sub_label=sub_label,
        verbose=False,
    )
    triton_comparable_results = [m for _, m in triton_benchmark_results]

    ref_benchmark_results = benchmark_all(
        ref_fn,
        q,
        k,
        v,
        bias,
        causal,
        dropout_p,
        repeats=WARMUP_REPS,
        desc="Standard Attention",
        sub_label=sub_label,
        verbose=False,
    )
    ref_comparable_results = [m for _, m in ref_benchmark_results]

    return triton_comparable_results + ref_comparable_results


if __name__ == "__main__":
    torch.manual_seed(0)

    dtype = torch.bfloat16

    all_results = []
    for batch_size, nheads, seqlen, d in [(1, 12, 1024, 128), (1, 12, 2048, 128)]:
        for dropout_p in [0.0, 0.1]:
            for causal in [False, True]:
                comparable_results = run_benchmark(
                    batch_size=batch_size,
                    n_heads=nheads,
                    seq_len=seqlen,
                    head_dim=d,
                    causal=causal,
                    dropout_p=dropout_p,
                    dtype=dtype,
                )
                all_results.extend(comparable_results)

    compare = Compare(all_results)
    compare.colorize(rowwise=True)
    compare.print()
