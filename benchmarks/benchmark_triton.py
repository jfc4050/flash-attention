from argparse import ArgumentParser
import math
import torch
from torch.utils.benchmark import Compare
from benchmarks.attn_fused_softmax import attn_fused_softmax

from flash_attn.utils.benchmark import benchmark_all
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.flash_attn_triton import flash_attn_func

WARMUP_REPS = 30


def attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if bias is not None:
        raise NotImplementedError

    batch_sz, seqlen, nheads, head_dim = q.shape

    cu_seqlens = torch.arange(
        0,
        (batch_sz + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=q.device
    )
    return flash_attn_unpadded_func(
        q.view(-1, nheads, head_dim),
        k.view(-1, nheads, head_dim),
        v.view(-1, nheads, head_dim),
        cu_seqlens,
        cu_seqlens,
        seqlen,
        seqlen,
        dropout_p=dropout_p,
        softmax_scale=None,
        causal=causal,
        return_attn_probs=False
    )

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
    implementations_to_benchmark: set,
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    bias_shape: tuple,
    causal: bool,
    dropout_p: float,
) -> list:
    device = "cuda"
    dtype = torch.bfloat16

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
    if bias_shape is None:
        bias = None
    else:
        bias = torch.randn(
            bias_shape,
            requires_grad=False,
            device=device,
            dtype=dtype,
        )

    # run benchmarks
    sub_label = f"({batch_size},{n_heads},{seq_len},{head_dim}), p={dropout_p}, bias={bias is not None}, causal={causal}"
    results = []

    if "cuda" in implementations_to_benchmark:
        try:
            cuda_fn = lambda q, k, v, bias, causal, dropout_p: attention_cuda(
                q, k, v, bias, causal, dropout_p
            )
            cuda_benchmark_results = benchmark_all(
                cuda_fn,
                q,
                k,
                v,
                bias,
                causal,
                dropout_p,
                repeats=WARMUP_REPS,
                desc="Flash (CUDA)",
                sub_label=sub_label,
                verbose=False
            )
            results.extend([m for _, m in cuda_benchmark_results])
        except:
            pass

    if "triton" in implementations_to_benchmark:
        triton_fn = lambda q, k, v, bias, causal, dropout_p: flash_attn_func(
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
            desc="Flash (Triton)",
            sub_label=sub_label,
            verbose=False,
        )
        results.extend([m for _, m in triton_benchmark_results])

    if "fused_softmax" in implementations_to_benchmark:
        try:
            bhmk_q = q.permute(0, 2, 1, 3)
            bhmk_k = k.permute(0, 2, 1, 3)
            bhmk_v = v.permute(0, 2, 1, 3)
            fused_softmax_fn = lambda q, k, v, bias, causal, dropout_p: attn_fused_softmax(
                bhmk_q, bhmk_k, bhmk_v, bias, causal, dropout_p
            )
            fused_softmax_results = benchmark_all(
                fused_softmax_fn,
                q,
                k,
                v,
                bias,
                causal,
                dropout_p,
                repeats=WARMUP_REPS,
                desc="Fused Softmax Attention",
                sub_label=sub_label,
                verbose=False,
            )
            results.extend([m for _, m in fused_softmax_results])
        except:
            pass

    if "ref" in implementations_to_benchmark:
        try:
            ref_fn = lambda q, k, v, bias, causal, dropout_p: attention_ref(
                q, k, v, bias, causal, dropout_p
            )
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
            results.extend([m for _, m in ref_benchmark_results])
        except:
            pass

    return results


if __name__ == "__main__":
    parser = ArgumentParser(__doc__)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    implementations_to_benchmark = {"cuda", "triton", "ref"}
    if args.all:
        implementations_to_benchmark = {"cuda", "triton", "fused_softmax", "ref"}
    else:
        implementations_to_benchmark = {"cuda", "triton", "ref"}

    torch.manual_seed(0)

    all_results = []
    for batch_size, nheads, seqlen, d in [(1, 12, 4096, 128)]:
        for bias_shape in [None, (batch_size, 1, 1, seqlen)]:
            for dropout_p in [0.0, 0.1]:
                for causal in [False, True]:
                    comparable_results = run_benchmark(
                        implementations_to_benchmark=implementations_to_benchmark,
                        batch_size=batch_size,
                        n_heads=nheads,
                        seq_len=seqlen,
                        head_dim=d,
                        bias_shape=bias_shape,
                        causal=causal,
                        dropout_p=dropout_p,
                    )
                    all_results.extend(comparable_results)

    compare = Compare(all_results)
    compare.colorize(rowwise=True)
    compare.print()
