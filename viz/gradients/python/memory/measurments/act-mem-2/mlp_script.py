"""
Prints out the ratio of activation memory for the MLP layer when using ReLU vs GELU.
"""

import torch
import torch.nn as nn

import act_mem
import layers

if __name__ == "__main__":
    batch_size, seq_len, d_model, dropout_prob = 1, 128, 1024, 0.1
    print(f"Batch size: {batch_size}, sequence length: {seq_len}, d_model: {d_model}, dropout_prob: {dropout_prob}  ")
    dtype = torch.bfloat16
    inputs = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device="cuda",
        requires_grad=True,
        dtype=dtype,
    )

    act_fn_dict = {"ReLU": nn.ReLU() , "GELU": nn.GELU(), "silu": nn.SiLU()}
    # Append outputs to a list to keep tensors alive
    outputs = []
    mem_bytes = []

    for name, act_fn in act_fn_dict.items():
        if name == "silu":
            mlp = layers.SwiGLUMLP(
                d_model=d_model,
                intermediate_size=4 * d_model,
                act_fn=act_fn,
                dropout_prob=dropout_prob,
                device="cuda",
                dtype=dtype,
            )
        else:
            mlp = layers.MLP(
                d_model=d_model,
                act_fn=act_fn,
                dropout_prob=dropout_prob,
                device="cuda",
                dtype=dtype,
        )
        with act_mem.AllocatedMemContext() as mem, act_mem.SavedTensorContext(
            ignored_tensors=mlp.parameters()
        ) as saved:
            out = mlp(inputs)
            outputs.append(out)
        stm = saved.saved_tensor_mem
        assert mem.delta["current"] == stm
        print(f"{name} bytes: {act_mem.B_to_GiB(stm)}")
        mem_bytes.append(stm)

    print(f"ReLU/GELU act mem ratio: {mem_bytes[0]/mem_bytes[1]}")
