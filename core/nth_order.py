from itertools import product, combinations
from heapq import heapify, heappop, heappush
from dataclasses import dataclass, field
from typing import Iterable
from tqdm import tqdm

import torch as th
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.surgical_olmo import SurgicalOlmo2ForCausalLM


@dataclass(order=True)
class NthOrderDelta:
    unit_idx: int
    delta: th.Tensor | None = field(default=None, compare=False)
    parent: "NthOrderDelta | None" = field(default=None, compare=False)
    children: list["NthOrderDelta"] = field(default_factory=list, compare=False)

    def __getitem__(self, indices) -> "NthOrderDelta":
        if not isinstance(indices, tuple):
            indices = (indices,)

        idx, *rest = indices
        absolute_idx = idx - self.unit_idx - 1

        assert 0 <= absolute_idx < len(self.children), f"Index {idx} out of bounds for unit {self.unit_idx}"

        if not rest:
            return self.children[absolute_idx]
        else:
            return self.children[absolute_idx][rest]

    def __setitem__(self, indices, value: "NthOrderDelta") -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)

        idx, *rest = indices
        absolute_idx = idx - self.unit_idx - 1

        assert 0 <= absolute_idx < len(self.children), f"Index {idx} out of bounds for unit {self.unit_idx}"

        if not rest:
            self.children[absolute_idx] = value
        else:
            self.children[absolute_idx][rest] = value

    def unit_indices(self) -> list[int]:
        current_node = self
        indices = []

        while current_node is not None:
            indices = [current_node.unit_idx] + indices
            current_node = current_node.parent

        return indices[1:]

def compute_unit_jacobians_and_outputs(model: SurgicalOlmo2ForCausalLM, inputs: dict) -> tuple[Iterable[Iterable[tuple[int, int, th.Tensor, th.Tensor]]], th.Tensor]:
    input_embeddings = model.get_input_embeddings()
    inputs_embeds = input_embeddings(inputs["input_ids"].to(model.device)).detach()
    inputs_embeds.requires_grad = True
    attention_mask = inputs["attention_mask"]  # B, T

    position_ids = th.arange(
        0, inputs_embeds.shape[1], device=inputs_embeds.device, dtype=th.long
    )
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids.unsqueeze(0))
    causal_mask = model.model._update_causal_mask(
        None, inputs_embeds, position_ids, None, True
    )

    unit_forwards = model.model.unit_forwards()

    # try to clear up memory by only loading the necessary parts of the model
    del model
    th.cuda.empty_cache()

    batch_size, seq_len, hidden_size = inputs_embeds.shape

    def unit_jacobian_output_generator():
        for unit_idx, unit_forward in enumerate(unit_forwards):
            if unit_idx % 2 == 0:  # attention unit
                output = unit_forward(
                    hidden_states=inputs_embeds,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    track_activations=False,
                )
                jacobian = th.empty(hidden_size, hidden_size, device=inputs_embeds.device)

                def jacobian_output_generator():
                    for batch_idx, seq_idx in attention_mask.nonzero():
                        for i in tqdm(range(hidden_size), desc=f"Computing jacobian for unit {unit_idx}", leave=False, total=hidden_size):
                            grad_mask = th.zeros_like(output)
                            grad_mask[batch_idx, seq_idx, i] = 1

                            jacobian[i] = th.autograd.grad(
                                output,
                                inputs_embeds,
                                grad_outputs=grad_mask,
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=True,
                            )[0][batch_idx, seq_idx]

                        yield batch_idx, seq_idx, jacobian, output[batch_idx, seq_idx]
            else:  # mlp unit
                output = unit_forward(
                    hidden_states=inputs_embeds,
                    track_activations=False,
                )

                def jacobian_output_generator():
                    for batch_idx, seq_idx in attention_mask.nonzero():
                        jacobian = th.autograd.functional.jacobian(
                            unit_forward,
                            inputs_embeds[batch_idx, seq_idx],
                        )

                        yield batch_idx, seq_idx, jacobian, output[batch_idx, seq_idx]

            yield jacobian_output_generator()

    return unit_jacobian_output_generator(), inputs_embeds

def compute_nth_order_deltas(
    model: SurgicalOlmo2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    stop_n: int = 2,
    max_token_length: int = 512,
) -> tuple[NthOrderDelta, th.Tensor, dict]:
    """Compute up to the max_nth order deltas for the given model and dataset.

    Args:
        model (SurgicalOlmo2ForCausalLM): The model to compute the nth order deltas for.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the dataset.
        dataset (Dataset): The dataset to compute the nth order deltas for.
        stop_n (int): The maximum order of the deltas to compute. This is exclusive, from 0 to stop_n - 1.

    Returns:
        NthOrderDelta: The tree of nth order deltas for the given model and dataset.
        th.Tensor: The final hidden state of the model before the final norm and output layer.
        dict: The inputs to the model.
    """
    assert stop_n > 0, "stop_n must be greater than 0"

    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
    inputs["attention_mask"] = inputs["attention_mask"].bool()  # why tf is this an int64
    inputs["input_ids"] = inputs["input_ids"].to(model.device)

    labels = th.full_like(inputs["input_ids"], -100)
    labels[inputs["attention_mask"]] = inputs["input_ids"][inputs["attention_mask"]]
    inputs["labels"] = labels

    # the output of unit f_i is f_i(f_{i-1}(f_{i-2}(...(f_0(x) + x)...) + x) + f_{i-2}(...) + ... + x)
    # we want to linearly separate the terms so that we can get the contributions of each one
    # we do this by approximating with a taylor series expansion using the jacobian as follows
    #
    #    f_i(x + h) ≈ f_i(x) + J_i(x)h
    #
    # => f_i(f_{i-1}(... + x) + f_{i-2}(...) + ... + x) ≈ J_i(x)(f_{i-1}(... + x) + f_{i-2}(...) + ... + f_0(x)) + f_i(x)
    #                                                   ≈ f_i(x) + J_i(x)f_0(x) + J_i(x)(f_1(f_0(x) + x)) + ...
    #                                                   ≈ f_i(x) + J_i(x)f_0(x) + J_i(x)J_1(x)f_0(x) + J_i(x)f_1(x)
    # in order, the above RHS gives us the first order effect of f_i, the second order effect of f_i and f_0, the third order effect of f_i, f_0, and f_1, the second order effect of f_i and f_1, and so on
    # in general, the nth order effect of f_{i_0}, f_{i_1}, ..., f_{i_n} is given by
    #
    #    J_{i_n}(x)J_{i_{n-1}}(x)...J_{i_1}(x)f_{i_0}(x)

    num_units = len(model.model.unit_forwards())
    total_iterations = num_units * inputs["attention_mask"].sum().item()

    jacobians_and_outputs, base = compute_unit_jacobians_and_outputs(model, inputs)
    final_residual = model(
        **inputs,
        activation_mask=["model_activations.layer_activations.63.output"],
    ).model_activations.layer_activations[-1].output

    zeroth_order_delta, units_deltas = empty_nth_order_deltas_recursive(delta=base, num_units=num_units, max_depth=stop_n)

    with tqdm(total=total_iterations, desc="Computing nth order deltas") as progress_bar:
        for unit_deltas, jacobian_output_generator in zip(units_deltas, jacobians_and_outputs):
            for batch_idx, seq_idx, jacobian, output in jacobian_output_generator:
                for unit_delta in unit_deltas:
                    if unit_delta.delta is None:
                        unit_delta.delta = th.empty_like(base)

                    if unit_delta.parent is zeroth_order_delta:
                        unit_delta.delta[batch_idx, seq_idx] = output
                    else:
                        # because we go through this in order of units, the parent is guaranteed to be computed
                        # since the parent's unit_idx < current unit_idx
                        unit_delta.deltas[batch_idx, seq_idx] = jacobian @ unit_delta.parent.delta[batch_idx, seq_idx]

                    progress_bar.update(1)

    return zeroth_order_delta, final_residual, inputs


# hehe dfs
def empty_nth_order_deltas_recursive(
    delta: th.Tensor | None = None,
    depth: int = 0,
    unit_idx: int = -1,
    num_units: int = 64,
    unit_deltas: list[list[NthOrderDelta]] | None = None,
    max_depth: int = 1,
) -> tuple[NthOrderDelta, list[list[NthOrderDelta]]]:
    if depth == 0:
        assert delta is not None, "base must be provided if depth is 0"
        assert unit_deltas is None, "unit_deltas must be None if depth is 0"
        assert unit_idx == -1, "unit_idx must be -1 if depth is 0"

        nth_order_delta = NthOrderDelta(unit_idx=-1, delta=delta)
        unit_deltas = [[] for _ in range(num_units)]
        delta = None
    else:
        assert delta is None, "base must be None if depth is not 0"
        assert unit_deltas is not None, "unit_deltas must be provided if depth is not 0"
        assert unit_idx > -1, "unit_idx must be non-negative if depth is not 0"

        nth_order_delta = NthOrderDelta(unit_idx=unit_idx)
        unit_deltas[unit_idx].append(nth_order_delta)

    if depth >= max_depth:
        return nth_order_delta, unit_deltas

    for new_unit_idx in range(unit_idx + 1, num_units):
        new_nth_order_delta, _ = empty_nth_order_deltas_recursive(
            delta=delta,
            depth=depth + 1,
            unit_idx=new_unit_idx,
            num_units=num_units,
            unit_deltas=unit_deltas,
            max_depth=max_depth,
        )
        new_nth_order_delta.parent = nth_order_delta
        nth_order_delta.children.append(new_nth_order_delta)

    return nth_order_delta, unit_deltas
