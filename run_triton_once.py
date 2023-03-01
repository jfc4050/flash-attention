import torch
from flash_attn.flash_attn_triton import flash_attn_func

batch_size = 1
seq_len = 2048
n_heads = 12
head_dim = 128
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
bias = None
causal = True
blockmask = torch.randn(seq_len // 128 ,seq_len // 128, requires_grad=False, device=device, dtype=dtype)
dropout_p = 0.0

out = flash_attn_func(q, k, v, bias, causal, blockmask, dropout_p)

grad_out = torch.rand_like(q)
out.backward(grad_out)
