from dataclasses import dataclass, asdict
from functools import partial
from typing import Callable
import gc
import os
import yaml
import time

import arguably
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.nth_order import compute_nth_order_deltas, NthOrderDelta
from exp.exp_data import DTYPE_MAP


@dataclass
class ReconstructionError:
    l1: th.Tensor
    l2: th.Tensor
    cosine: th.Tensor


@arguably.command
def main(
    *args,
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    maxlen: int = 512,
    device: str = "cuda",
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 3,
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer, checkpoint = get_model_and_tokenizer(model_name, checkpoint_idx, model_kwargs=model_kwargs)

    deltas, depth_deltas, units_deltas, final_state, inputs = compute_nth_order_deltas(model, checkpoint, tokenizer, dataset, stop_n=n, max_token_length=maxlen)

    with th.no_grad():
        reconstructed_activations = reconstruct_from_deltas(deltas.delta, units_deltas, n)

    del dataset, model_kwargs, tokenizer, deltas, depth_deltas, units_deltas, final_state
    th.cuda.empty_cache()
    gc.collect()

    reconstruction_errors = [None] * n

    if "pythia" in model_name:
        activation_mask = [
            "model_activations.layer_activations.*.output"
        ]
        activations = model(**inputs, activation_mask=activation_mask, track_activations=True)
        layer_activations = activations.model_activations.layer_activations

        for layer_idx, layer_activation in enumerate(layer_activations[:n]):
            ground_truth = layer_activation.output
            reconstructed = reconstructed_activations[layer_idx].to(ground_truth.device)
            ground_truth = ground_truth.view(-1, ground_truth.shape[-1])
            reconstructed = reconstructed.view(-1, reconstructed.shape[-1])

            l1 = F.pairwise_distance(ground_truth, reconstructed, p=1).mean()
            l2 = F.pairwise_distance(ground_truth, reconstructed, p=2).mean()
            cosine = F.cosine_similarity(ground_truth, reconstructed).mean()

            reconstruction_errors[layer_idx] = ReconstructionError(l1=l1, l2=l2, cosine=cosine)
    else:
        raise NotImplementedError(f"Model {model_name} not supported")

    for layer_idx, reconstruction_error in enumerate(reconstruction_errors):
        print(f"Layer {layer_idx} reconstruction error:")
        print(f"        L1: {reconstruction_error.l1.item()}")
        print(f"        L2: {reconstruction_error.l2.item()}")
        print(f"    Cosine: {reconstruction_error.cosine.item()}")
        print()

def reconstruct_from_deltas(
    base_residual: th.Tensor,
    units_deltas: list[list[NthOrderDelta]],
    n: int,
) -> list[th.Tensor]:
    reconstructed_activations = [None] * n
    current_residual = base_residual.clone()

    for unit_idx, unit_deltas in enumerate(units_deltas[:n]):
        for delta in unit_deltas:
            current_residual += delta.delta.to(current_residual.device)

        reconstructed_activations[unit_idx] = current_residual.clone()

    return reconstructed_activations


if __name__ == "__main__":
    arguably.run()
