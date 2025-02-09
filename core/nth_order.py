from itertools import product, combinations

import torch as th
from torch.func import vmap, jacrev, functional_call
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.surgical_olmo import SurgicalOlmo2ForCausalLM


def empty_nth_order_deltas(num_units: int, max_n: int) -> list:
    nth_order_deltas = [None] * max_n

    # create a list of list of ... of None of dimension num_units x num_units x ...
    # make this n layers deep
    for n in range(max_n):
        for _ in range(n):
            current_hypercube = nth_order_deltas[n]
            nth_order_deltas[n] = [current_hypercube.copy() for _ in range(num_units)]

    return nth_order_deltas

def compute_jacobians(model: SurgicalOlmo2ForCausalLM, inputs: dict) -> list:
    input_embeddings = model.get_input_embeddings()
    inputs_embeds = input_embeddings(inputs["input_ids"])
    attention_mask = inputs["attention_mask"]  # B, T

    unit_forwards = model.model.unit_forwards()
    U = len(unit_forwards)

    B, T, D = inputs_embeds.shape

    jacobians = th.empty(U, B, T, D, D, device=model.device)

    # batched jacobians for mlp units
    for unit_idx in range(1, U, 2):
        unit_forward = unit_forwards[unit_idx]
        debatched_inputs_embeds = inputs_embeds[attention_mask]  # T', D where T' is all the tokens across all batches

        def functional_unit_forward(inputs_embeds):
            return functional_call(unit_forward, inputs_embeds)

        debatched_jacobians = vmap(jacrev(functional_unit_forward))  # T', D, D
        jacobians[attention_mask.unsqueeze(0)] = debatched_jacobians(debatched_inputs_embeds)

    # we have to do it this way for attention because we cant batch by sequence length
    # meaning we would end up with a massive inefficient block diagonal matrix
    for unit_idx, (batch_idx, seq_idx) in product(range(0, U, 2), attention_mask.nonzero()):
        unit_forward = unit_forwards[unit_idx]
        input_embed = inputs_embeds[batch_idx, :seq_idx + 1].unsqueeze(0)  # 1, S, D where S is the sequence length based on the index of this token
        output_embed = unit_forward(input_embed)  # 1, S, D

        for i in range(D):
            jacobians[unit_idx, batch_idx, seq_idx, i] = th.autograd.grad(
                output_embed[0, -1, i],
                input_embed[0, -1],
                grad_outputs=th.ones_like(output_embed[0, -1, i]),
                retain_graph=True,
                create_graph=True,
            )[0]

    return jacobians

def compute_nth_order_deltas(
    model: SurgicalOlmo2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    max_n: int,
) -> list:
    """Compute up to the max_nth order deltas for the given model and dataset.

    Args:
        model (SurgicalOlmo2ForCausalLM): The model to compute the nth order deltas for.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the dataset.
        dataset (Dataset): The dataset to compute the nth order deltas for.
        max_n (int): The maximum order of the deltas to compute.

    Returns:
        list: The nth order deltas for the model and dataset.
    """
    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)

    batch_size, seq_len = inputs["input_ids"].shape
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_units = num_layers * 2

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

    nth_order_deltas = empty_nth_order_deltas(num_units, max_n)
    jacobians = compute_jacobians(model, inputs)

    for i in range(num_units):
        # TODO: compute deltas using jacobians
        pass
