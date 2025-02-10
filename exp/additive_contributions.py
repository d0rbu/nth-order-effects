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

@th.no_grad()
@arguably.command
def main(
    *args,
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    n: int = 1,
) -> None:
    dataset = get_dataset(dataset_name)
    model, tokenizer = get_model_and_tokenizer(checkpoint_idx)

    deltas, final_state, inputs = compute_nth_order_deltas(model, tokenizer, dataset, n=n)
    null_input = th.zeros_like(deltas.delta)

    loss_fn = partial(model.loss_function, labels=inputs["labels"], vocab_size=model.config.vocab_size)

    output_norm = model.model.norm
    lm_head = model.get_output_embeddings()
    output_module = nn.Sequential(output_norm, lm_head)

    with th.no_grad():
        delta_losses = compute_losses(null_input, deltas, loss_fn, output_module)

    sorted_delta_losses = sorted(delta_losses, key=lambda x: x.loss)

    # TODO: plot a bar chart of the delta losses, labelling by delta_loss.nth_order_delta.unit_indices()

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
