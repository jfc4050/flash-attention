from flash_attn.layers.rotary import apply_rotary_emb_torch, apply_rotary_emb_func
import torch
from torch.utils.benchmark import Compare

from flash_attn.utils.benchmark import benchmark_all

WARMUP_REPS = 30


results = []
dtype = torch.bfloat16
for batch_size, nheads, seqlen, d in [(1, 12, 4096, 128)]:
    for rotary_fraction in [0.5, 1.0]:
        for inplace in [True, False]:
            sub_label = f"({batch_size}, {nheads}, {seqlen}, {d}), f={rotary_fraction}, inplace={inplace}"
            x = torch.randn(
                batch_size,
                seqlen,
                nheads,
                d,
                dtype=dtype,
                device="cuda",
                requires_grad=True,
            )

            rotary_dim = int(rotary_fraction * d)
            angle = torch.randn(seqlen, rotary_dim // 2, device="cuda")
            cos = torch.cos(angle).to(dtype=dtype)
            sin = torch.sin(angle).to(dtype=dtype)

            flash_fn = lambda x, cos, sin, inplace: apply_rotary_emb_func(
                x, cos, sin, inplace
            )
            flash_bench_results = benchmark_all(
                flash_fn,
                x,
                cos,
                sin,
                inplace,
                repeats=WARMUP_REPS,
                desc="flash rotary",
                sub_label=sub_label,
                verbose=False,
            )
            results.extend([m for _, m in flash_bench_results])

            if not inplace:
                torch_fn = lambda x, cos, sin: apply_rotary_emb_torch(x, cos, sin)

                torch_bench_results = benchmark_all(
                    torch_fn,
                    x,
                    cos,
                    sin,
                    repeats=WARMUP_REPS,
                    desc="torch rotary",
                    sub_label=sub_label,
                )
compare = Compare(t results)
compare.colorize(rowwise=True)
compare.print()

