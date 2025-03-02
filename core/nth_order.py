from functools import partial
from dataclasses import dataclass, field
from typing import Iterable
import json
import hashlib
from math import comb

import torch as th
import torch.nn as nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tqdm import tqdm
from cachier import cachier

from core.model import SurgicalModel, Checkpoint
from core.surgical_gpt_neox import SurgicalGPTNeoXForCausalLM
from core.surgical_olmo import SurgicalOlmo2ForCausalLM


@dataclass
class NthOrderDelta:
    unit_idx: int
    delta: th.Tensor | None = None
    parent: "NthOrderDelta | None" = None
    children: list["NthOrderDelta"] = field(default_factory=list)

    def __getitem__(self, indices: tuple[int] | int) -> "NthOrderDelta":
        if not isinstance(indices, tuple):
            indices = (indices,)

        idx, *rest = indices
        absolute_idx = idx - self.unit_idx - 1

        assert 0 <= absolute_idx < len(self.children), f"Index {idx} out of bounds for unit {self.unit_idx}"

        if not rest:
            return self.children[absolute_idx]
        else:
            return self.children[absolute_idx][rest]

    def __setitem__(self, indices: tuple[int] | int, value: "NthOrderDelta") -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)

        idx, *rest = indices
        absolute_idx = idx - self.unit_idx - 1

        assert 0 <= absolute_idx < len(self.children), f"Index {idx} out of bounds for unit {self.unit_idx}"

        if not rest:
            self.children[absolute_idx] = value
        else:
            self.children[absolute_idx][rest] = value

    def __repr__(self) -> str:
        if self.delta is None:
            delta_repr = "None"
        else:
            delta_repr = f"Tensor(shape={self.delta.shape}, dtype={self.delta.dtype})"

        return f"NthOrderDelta(unit_indices={self.unit_indices()}, delta={delta_repr})"

    def unit_indices(self) -> list[int]:
        current_node = self
        indices = []

        while current_node is not None:
            indices = [current_node.unit_idx] + indices
            current_node = current_node.parent

        return indices[1:]

def cache_hash(
    args: tuple[SurgicalModel, PreTrainedTokenizerBase, list[str], int, int],
    kwargs: dict[str, int],
) -> str:
    model = args[0] if len(args) > 0 else kwargs.get("model")
    checkpoint = args[1] if len(args) > 1 else kwargs.get("checkpoint")
    tokenizer = args[2] if len(args) > 2 else kwargs.get("tokenizer")
    dataset = args[3] if len(args) > 3 else kwargs.get("dataset")
    stop_n = args[4] if len(args) > 4 else kwargs.get("stop_n")
    max_token_length = args[5] if len(args) > 5 else kwargs.get("max_token_length")

    checkpoint_str = "latest" if checkpoint is None else f"{checkpoint.step}_{checkpoint.num_tokens}"
    hf_config_str = json.dumps(model.config.to_dict(), sort_keys=True)
    dataset_str = "_".join(dataset)
    stop_n_str = str(stop_n)
    max_token_length_str = str(max_token_length)

    hashes = [hashlib.sha256(arg.encode()).hexdigest() for arg in [checkpoint_str, hf_config_str, dataset_str, stop_n_str, max_token_length_str]]

    return "_".join(hashes)

def compute_nth_order_deltas_backward(
    model: SurgicalModel,
    checkpoint: Checkpoint | None,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    stop_n: int = 2,
    max_token_length: int = 512,
) -> tuple[NthOrderDelta, list[list[NthOrderDelta]], list[list[NthOrderDelta]], dict, list[th.Tensor]]:
    """Compute up to the max_nth order deltas for the given model and dataset. This function uses backpropagation to compute the nth order deltas."""
    assert stop_n > 0, "stop_n must be greater than 0"

    # make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    inputs["input_ids"] = inputs["input_ids"].to(model.device)

    labels = th.full_like(inputs["input_ids"], -100)
    labels[inputs["attention_mask"]] = inputs["input_ids"][inputs["attention_mask"]]
    inputs["labels"] = labels

    num_units = len(model.model.unit_forwards())

    match model:
        case SurgicalGPTNeoXForCausalLM():
            activations = model(
                **inputs,
                activation_mask=["model_activations.layer_activations.*.output", "loss", "model_activations.residual_base"],
            )
            inputs_embeds = activations.model_activations.residual_base
            layer_activations = [inputs_embeds] + [layer_activation.output for layer_activation in activations.model_activations.layer_activations]
        case SurgicalOlmo2ForCausalLM():
            raise NotImplementedError("SurgicalOlmo2ForCausalLM is not yet supported")

    gradients = [
        th.autograd.grad(
            activations.loss,
            layer_activation,
            retain_graph=True,
        )[0]
        for layer_activation in layer_activations
    ]

    del activations, inputs_embeds

    # zeroth_order_delta is the root node, depth_deltas is the deltas in a list ordered by depth, and units_deltas is the deltas in a list ordered by the last unit index
    zeroth_order_delta, depth_deltas, units_deltas = empty_nth_order_deltas_recursive(delta=gradients[-1].cpu(), num_units=num_units, max_depth=stop_n)

    total_iterations = sum(len(unit_deltas) for unit_deltas in units_deltas)

    with tqdm(total=total_iterations, desc="Computing nth order gradient deltas", leave=False) as progress_bar:
        for inverse_unit_idx, (unit_deltas, unit, unit_input, unit_output) in enumerate(zip(units_deltas, reversed(model.model.unit_forwards()), reversed(layer_activations[:-1]), reversed(layer_activations[1:]))):
            unit_output_idx = num_units - inverse_unit_idx
            for unit_delta in unit_deltas:
                is_last = unit_delta is unit_deltas[-1]

                unit_delta.delta = th.autograd.grad(
                    unit_output,
                    unit_input,
                    grad_outputs=unit_delta.parent.delta.to(model.device),
                    retain_graph=not is_last,
                )[0].cpu()

                progress_bar.update(1)

            layer_activations[unit_output_idx] = None
            del unit, unit_input, unit_output
            th.cuda.empty_cache()

    return zeroth_order_delta, units_deltas, inputs, gradients


@cachier(cache_dir=".nth_order_delta_direct_cache", hash_func=cache_hash)
def compute_nth_order_deltas_direct(
    model: SurgicalModel,
    checkpoint: Checkpoint | None,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    stop_n: int = 2,
    max_token_length: int = 512,
) -> tuple[NthOrderDelta, list[list[NthOrderDelta]], list[list[NthOrderDelta]], th.Tensor, dict]:
    """Compute up to the max_nth order deltas for the given model and dataset.

    Args:
        model (SurgicalModel): The model to compute the nth order deltas for.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for encoding the dataset.
        dataset (Dataset): The dataset to compute the nth order deltas for.
        stop_n (int): The maximum order of the deltas to compute. This is exclusive, from 0 to stop_n - 1.

    Returns:
        NthOrderDelta: The tree of nth order deltas for the given model and dataset.
        list[list[NthOrderDelta]]: The nth order deltas in a list ordered by depth.
        list[list[NthOrderDelta]]: The nth order deltas in a list ordered by the last unit index.
        th.Tensor: The final residual of the model.
        dict: The inputs used to compute the nth order deltas.
    """
    assert stop_n > 0, "stop_n must be greater than 0"

    # make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    inputs["input_ids"] = inputs["input_ids"].to(model.device)

    labels = th.full_like(inputs["input_ids"], -100)
    labels[inputs["attention_mask"]] = inputs["input_ids"][inputs["attention_mask"]]
    inputs["labels"] = labels

    input_embeddings = model.get_input_embeddings()
    inputs_embeds = input_embeddings(inputs["input_ids"].to(model.device)).detach()

    # the output of unit f_0 is f_0(x) and includes the first order effect of f_0
    # the output of unit f_1 is f_1(f_0(x) + x) and includes the first order effect of f_1 and the second order effect of f_0 and f_1
    # to separate these, we consider the first order effect of f_1 to be f_1(x) and the second order effect to be f_1(f_0(x) + x) - f_1(x)
    # the output of unit f_2 is f_2(f_1(f_0(x) + x) + f_0(x) + x) and includes four effects of order 1, 2, 2, and 3
    # to separate these, we consider the first order effect of f_2 to be f_2(x), the second order effects of f_0 and f_2 to be f_2(f_0(x) + x) - f_2(x),
    # the second order effects of f_1 and f_2 to be f_2(f_1(x) + x) - f_2(x), and the third order effect of f_0, f_1, and f_2 to be f_2(f_1(f_0(x) + x) + f_0(x) + x) - f_2(f_1(x) + x) - f_2(f_0(x) + x) + f_2(x)
    # f_2(f_1(f_0(x) + x) + f_0(x) + x) - f_2(f_0(x) + f_1(x) + x)

    num_units = len(model.model.unit_forwards())
    num_layers = model.config.num_hidden_layers
    with th.no_grad():
        final_residual = model(
            **inputs,
            activation_mask=[f"model_activations.layer_activations.{num_layers - 1}.output"],
        ).model_activations.layer_activations[-1].output

    # zeroth_order_delta is the root node, depth_deltas is the deltas in a list ordered by depth, and units_deltas is the deltas in a list ordered by the last unit index
    zeroth_order_delta, depth_deltas, units_deltas = empty_nth_order_deltas_recursive(num_units=num_units, max_depth=stop_n)

    raise NotImplementedError("Direct computation of nth order deltas is not yet implemented")

# logic for computing from jacobians

@cachier(cache_dir=".nth_order_delta_jacobian_cache", hash_func=cache_hash)
def compute_nth_order_deltas_jacobian(
    model: SurgicalModel,
    checkpoint: Checkpoint | None,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    stop_n: int = 2,
    max_token_length: int = 512,
) -> tuple[NthOrderDelta, list[list[NthOrderDelta]], list[list[NthOrderDelta]], th.Tensor, dict]:
    """Compute up to the max_nth order deltas for the given model and dataset. This function uses jacobians to compute the nth order deltas."""
    assert stop_n > 0, "stop_n must be greater than 0"

    # make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    num_layers = model.config.num_hidden_layers
    total_iterations = num_units * inputs["attention_mask"].sum().item()

    jacobians_and_outputs, base = compute_unit_jacobians_and_outputs(model, inputs)
    with th.no_grad():
        final_residual = model(
            **inputs,
            activation_mask=[f"model_activations.layer_activations.{num_layers - 1}.output"],
        ).model_activations.layer_activations[-1].output

    # zeroth_order_delta is the root node, depth_deltas is the deltas in a list ordered by depth, and units_deltas is the deltas in a list ordered by the last unit index
    zeroth_order_delta, depth_deltas, units_deltas = empty_nth_order_deltas_recursive(delta=base, num_units=num_units, max_depth=stop_n)

    with tqdm(total=total_iterations, desc="Computing nth order deltas", leave=False) as progress_bar:
        for unit_deltas, jacobian_output_generator in zip(units_deltas, jacobians_and_outputs):
            for batch_idx, seq_idx, jacobian, output in jacobian_output_generator:
                with th.no_grad():
                    for unit_delta in unit_deltas:
                        if unit_delta.delta is None:
                            unit_delta.delta = th.empty_like(base, device=th.device("cpu"))

                        if unit_delta.parent is zeroth_order_delta:
                            unit_delta.delta[batch_idx, seq_idx] = output.to(th.device("cpu"))
                        else:
                            # because we go through this in order of units, the parent is guaranteed to be computed
                            # since the parent's unit_idx < current unit_idx
                            unit_delta.delta[batch_idx, seq_idx] = (jacobian @ unit_delta.parent.delta[batch_idx, seq_idx].to(jacobian.device)).to(th.device("cpu"))

                    del jacobian, output
                    th.cuda.empty_cache()

                    progress_bar.update(1)

    return zeroth_order_delta, depth_deltas, units_deltas, final_residual, inputs

def compute_unit_jacobians_and_outputs(model: SurgicalModel, inputs: dict) -> tuple[Iterable[Iterable[tuple[int, int, th.Tensor, th.Tensor]]], th.Tensor]:
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
    unit_forwards = [partial(unit_forward, track_activations=False) for unit_forward in unit_forwards]

    # try to clear up memory by only loading the necessary parts of the model
    use_parallel_residual = model.config.use_parallel_residual
    del model

    batch_size, seq_len, hidden_size = inputs_embeds.shape

    def unit_jacobian_output_generator():
        for unit_idx, unit_forward in enumerate(unit_forwards):
            if unit_idx % 2 == 0 or use_parallel_residual:
                output = unit_forward(
                    hidden_states=inputs_embeds,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                )
                jacobian = th.empty(hidden_size, hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

                def jacobian_output_generator():
                    nonlocal output
                    for batch_idx, seq_idx in attention_mask.nonzero():
                        for i in tqdm(range(hidden_size), desc=f"Computing jacobian for unit {unit_idx} batch {batch_idx} seq {seq_idx}", leave=False, total=hidden_size):
                            grad_mask = th.zeros_like(output)
                            grad_mask[batch_idx, seq_idx, i] = 1

                            jacobian[i] = th.autograd.grad(
                                output,
                                inputs_embeds,
                                grad_outputs=grad_mask,
                                retain_graph=True,
                            )[0][batch_idx, seq_idx]

                        yield batch_idx, seq_idx, jacobian, output[batch_idx, seq_idx]

                    del output

            else:  # mlp unit
                output = unit_forward(
                    hidden_states=inputs_embeds,
                )

                def jacobian_output_generator():
                    nonlocal output
                    for batch_idx, seq_idx in attention_mask.nonzero():
                        jacobian = th.autograd.functional.jacobian(
                            unit_forward,
                            inputs_embeds[batch_idx, seq_idx],
                        )

                        yield batch_idx, seq_idx, jacobian, output[batch_idx, seq_idx]
                        del jacobian

                    del output

            yield jacobian_output_generator()

    return unit_jacobian_output_generator(), inputs_embeds

# hehe dfs
def empty_nth_order_deltas_recursive(
    delta: th.Tensor | None = None,
    depth: int = 0,
    unit_idx: int = -1,
    num_units: int = 64,
    depth_deltas: list[list[NthOrderDelta]] | None = None,
    unit_deltas: list[list[NthOrderDelta]] | None = None,
    max_depth: int = 1,
) -> tuple[NthOrderDelta, list[list[NthOrderDelta]], list[list[NthOrderDelta]]]:
    if depth == 0:
        assert delta is not None, "base must be provided if depth is 0"
        assert unit_deltas is None, "unit_deltas must be None if depth is 0"
        assert unit_idx == -1, "unit_idx must be -1 if depth is 0"

        nth_order_delta = NthOrderDelta(unit_idx=-1, delta=delta)
        depth_deltas = [[nth_order_delta]] + [[] for _ in range(max_depth)]
        unit_deltas = [[] for _ in range(num_units)]
        delta = None
    else:
        assert delta is None, "base must be None if depth is not 0"
        assert unit_deltas is not None, "unit_deltas must be provided if depth is not 0"
        assert unit_idx > -1, "unit_idx must be non-negative if depth is not 0"

        nth_order_delta = NthOrderDelta(unit_idx=unit_idx)
        depth_deltas[depth].append(nth_order_delta)
        unit_deltas[unit_idx].append(nth_order_delta)

    if depth >= max_depth:
        return nth_order_delta, depth_deltas, unit_deltas

    for new_unit_idx in range(unit_idx + 1, num_units):
        new_nth_order_delta, _, _ = empty_nth_order_deltas_recursive(
            delta=delta,
            depth=depth + 1,
            unit_idx=new_unit_idx,
            num_units=num_units,
            depth_deltas=depth_deltas,
            unit_deltas=unit_deltas,
            max_depth=max_depth,
        )
        new_nth_order_delta.parent = nth_order_delta
        nth_order_delta.children.append(new_nth_order_delta)

    return nth_order_delta, depth_deltas, unit_deltas
