from itertools import product, combinations
from copy import deepcopy
from dataclasses import dataclass, field

import torch as th
from torch.func import vmap, jacrev, functional_call
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.surgical_olmo import SurgicalOlmo2ForCausalLM


@dataclass
class NthOrderDelta:
    delta: th.Tensor
    unit_idx: int
    parent: "NthOrderDelta" | None = None
    children: list["NthOrderDelta"] = field(default_factory=list)

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

def compute_unit_jacobians_and_outputs(model: SurgicalOlmo2ForCausalLM, inputs: dict) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    input_embeddings = model.get_input_embeddings()
    inputs_embeds = input_embeddings(inputs["input_ids"])
    attention_mask = inputs["attention_mask"]  # B, T

    unit_forwards = model.model.unit_forwards()
    num_units = len(unit_forwards)

    batch_size, seq_len, hidden_size = inputs_embeds.shape

    jacobians = th.empty(num_units, batch_size, seq_len, hidden_size, hidden_size, device=model.device)
    outputs = th.empty(num_units, batch_size, seq_len, hidden_size, device=model.device)

    # batched jacobians and forwards for mlp units
    for unit_idx in range(1, num_units, 2):
        unit_forward = unit_forwards[unit_idx]
        debatched_inputs_embeds = inputs_embeds[attention_mask]  # T', D where T' is all the tokens across all batches

        def functional_unit_forward(inputs_embeds):
            return functional_call(unit_forward, inputs_embeds)

        debatched_jacobians = vmap(jacrev(functional_unit_forward))  # T', D, D
        jacobians[unit_idx, attention_mask] = debatched_jacobians(debatched_inputs_embeds)
        outputs[unit_idx, attention_mask] = unit_forward(debatched_inputs_embeds)

    # we have to do it this way for attention because we cant batch by sequence length
    # meaning we would end up with a massive inefficient block diagonal matrix
    for unit_idx, (batch_idx, seq_idx) in product(range(0, num_units, 2), attention_mask.nonzero()):
        unit_forward = unit_forwards[unit_idx]
        input_embed = inputs_embeds[batch_idx, :seq_idx + 1].unsqueeze(0)  # 1, S, D where S is the sequence length based on the index of this token
        output_embed = unit_forward(input_embed)  # 1, S, D

        for i in range(hidden_size):
            jacobians[unit_idx, batch_idx, seq_idx, i] = th.autograd.grad(
                output_embed[0, -1, i],
                input_embed[0, -1],
                grad_outputs=th.ones_like(output_embed[0, -1, i]),
                retain_graph=True,
                create_graph=True,
            )[0]

        outputs[unit_idx, batch_idx, seq_idx] = output_embed[0, -1]

    return jacobians, outputs, inputs_embeds

def compute_nth_order_deltas(
    model: SurgicalOlmo2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    stop_n: int = 2,
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
    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)

    labels = th.full_like(inputs["input_ids"], -100)
    labels[input["attention_mask"]] = inputs["input_ids"][input["attention_mask"]]
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

    jacobians, outputs, base = compute_unit_jacobians_and_outputs(model, inputs)
    final_residual = model(
        **inputs,
        activation_mask=["model_activations.layer_activations.*.output"],
    ).model_activations.layer_activations[-1].output

    return compute_nth_order_deltas_recursive(
        jacobians,
        outputs,
        base,
        max_depth=stop_n,
        device=model.device,
    ), final_residual, inputs

# hehe bfs
def compute_nth_order_deltas_recursive(
    jacobians: th.Tensor,
    outputs: th.Tensor,
    base: th.Tensor,
    nth_order_delta: NthOrderDelta | None = None,
    depth: int = 0,
    max_depth: int = 1,
    device: th.device = th.device("cpu"),
) -> NthOrderDelta:
    assert depth >= 0, "Depth must be non-negative"
    assert max_depth >= 0, "Max depth must be non-negative"
    assert jacobians.shape[:2] == outputs.shape[:2], "Jacobian and output unit sizes and batch sizes must match"
    assert outputs.shape[1:] == base.shape, "Output and base unit sizes must match"

    num_units, batch_size, seq_len, hidden_size = outputs.shape

    if depth >= max_depth:
        return nth_order_delta
    elif depth == 0:
        zeroth_order_delta = NthOrderDelta(delta=base, unit_idx=-1)
        return compute_nth_order_deltas_recursive(jacobians, outputs, base, zeroth_order_delta, depth + 1, max_depth, device)

    current_unit_idx = nth_order_delta.unit_idx

    for unit_idx in range(current_unit_idx + 1, num_units):
        if depth == 1:
            new_delta = outputs[unit_idx]
        else:
            new_delta = jacobians[unit_idx] @ nth_order_delta.delta

        new_nth_order_delta = NthOrderDelta(delta=new_delta, unit_idx=unit_idx, parent=nth_order_delta)
        nth_order_delta.children.append(compute_nth_order_deltas_recursive(jacobians, outputs, base, new_nth_order_delta, depth + 1, max_depth, device))

    return nth_order_delta
