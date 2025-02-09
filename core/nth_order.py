from itertools import product

import torch as th
from datasets import Dataset, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from core.surgical_olmo import SurgicalOlmo2ForCausalLM


def nth_order_deltas(
    model: SurgicalOlmo2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    n: int = 1,
) -> th.Tensor:
    """Compute the nth order deltas for the given model and dataset.

    Args:
        model (SurgicalOlmo2ForCausalLM): The model to compute the nth order deltas for.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the dataset.
        dataset (Dataset): The dataset to compute the nth order deltas for.
        content_key (str): The key in the dataset to use as the content.

    Returns:
        th.Tensor: The nth order deltas for the given model and dataset.
    """
    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True)

    batch_size, seq_len = inputs["input_ids"].shape
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_units = num_layers * 2

    all_deltas = []

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

    for delta_order in range(0, n + 1):
        for units in product(range(num_units), repeat=delta_order):
            # ensure that the units are in strictly ascending order. essentially an upper triangular mask
            units_are_ascending = True
            for unit_idx in range(len(units) - 1):
                if units[unit_idx] >= units[unit_idx + 1]:
                    units_are_ascending = False
                    break

            if not units_are_ascending:
                continue

            # TODO: compute the current deltas for the given units
            current_deltas = th.empty(batch_size, seq_len, hidden_size, dtype=th.float32, device="cpu")
