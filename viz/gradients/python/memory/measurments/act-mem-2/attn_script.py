import torch

import act_mem
import layers

if __name__ == "__main__":
    batch_size, seq_len, d_model, n_heads = 1, 128, 1024, 32
    print(f"Batch size: {batch_size}, sequence length: {seq_len}, d_model: {d_model}, n_heads: {n_heads}")
    dtype = torch.bfloat16
    inputs = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    )

    attn = layers.Attention(
        d_model=d_model,
        n_heads=n_heads,
        device="cuda",
        dtype=dtype,
    )
    with act_mem.AllocatedMemContext() as mem, act_mem.SavedTensorContext(
        ignored_tensors=attn.parameters()
    ) as saved:
        out = attn(inputs)
    stm = saved.saved_tensor_mem
    print(f'{mem.delta["current"]=}')
    print(f"{stm=}")
    print(f"{stm/out.numel()=}")
