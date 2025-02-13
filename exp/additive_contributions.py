from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import arguably
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.nth_order import compute_nth_order_deltas, NthOrderDelta


@dataclass
class DeltaLoss:
    nth_order_delta: NthOrderDelta
    loss: float

DTYPE_MAP = {
    "bf16": th.bfloat16,
    "fp32": th.float32,
    "fp16": th.float16,
}

@th.no_grad()
@arguably.command
def main(
    *args,
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    maxlen: int = 512,
    device: str = "auto",
    dtype: str = "bf16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 2,
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer = get_model_and_tokenizer(checkpoint_idx, model_kwargs=model_kwargs)

    deltas, final_state, inputs = compute_nth_order_deltas(model, tokenizer, dataset, stop_n=n, max_token_length=maxlen)
    null_input = th.zeros_like(deltas.delta)

    loss_fn = partial(model.loss_function, labels=inputs["labels"], vocab_size=model.config.vocab_size)

    output_norm = model.model.norm
    lm_head = model.get_output_embeddings()
    output_module = nn.Sequential(output_norm, lm_head)

    with th.no_grad():
        delta_losses = compute_losses(null_input, deltas, loss_fn, output_module)

    sorted_delta_losses = sorted(delta_losses, key=lambda x: x.loss)

    plt.bar(range(len(sorted_delta_losses)), [x.loss for x in sorted_delta_losses], tick_label=[x.nth_order_delta.unit_indices() for x in sorted_delta_losses])

def compute_losses(
    base: th.Tensor,
    deltas: NthOrderDelta,
    loss_fn: Callable[[th.Tensor], th.Tensor],
    output_module: nn.Module,
) -> list[DeltaLoss]:
    delta_loss = loss_fn(output_module(base + deltas.delta)).item()

    losses = [DeltaLoss(deltas, delta_loss)]
    for child in deltas.children:
        losses.extend(compute_losses(base, child, loss_fn, output_module))

    return losses


if __name__ == "__main__":
    arguably.run()
