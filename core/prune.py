import torch as th
import gc
import torch.nn as nn
from typing import Callable, Self
from dataclasses import dataclass

from core.surgical_gpt_neox import SurgicalGPTNeoXForCausalLM
from core.surgical_olmo import SurgicalOlmo2ForCausalLM


@dataclass
class ModelUnit(frozen=True):
    block_idx: int
    unit_name: str

    def key(self) -> str:
        return f"{self.block_idx}_{self.unit_name}"


@dataclass
class ReplacedUnit(frozen=True):
    block_idx: int
    unit_name: str
    replacement_unit: nn.Module


ORDERED_MODEL_UNIT_CONFIGS = {
    SurgicalGPTNeoXForCausalLM: ["self_attn", "mlp"],
    SurgicalOlmo2ForCausalLM: ["attention", "mlp"],
}


def get_model_unit_key(model: SurgicalGPTNeoXForCausalLM | SurgicalOlmo2ForCausalLM) -> Callable[[], ModelUnit]:
    def key_fn(unit: ModelUnit) -> tuple[int, int]:
        unit_config = ORDERED_MODEL_UNIT_CONFIGS.get(type(model))
        assert unit_config is not None, f"Unsupported model type: {type(model)}"

        local_unit_idx = unit_config.index(unit.unit_name)
        return unit.block_idx, local_unit_idx

    return key_fn


def prune_model(model: SurgicalGPTNeoXForCausalLM | SurgicalOlmo2ForCausalLM, units_to_remove: set[ModelUnit] | ModelUnit) -> Callable[[], None]:
    """
    Prunes specified attention or MLP units from transformer model inplace, with hook to unprune.

    Args:
        model: The model instance.
        units_to_remove: A set of ModelUnit instances or a single ModelUnit instance to be removed.
    """
    if isinstance(units_to_remove, ModelUnit):
        units_to_remove = {units_to_remove}

    if len(units_to_remove) == 0:
        return lambda: None

    # Sort indices in descending order to avoid index shifting issues
    sorted_units = sorted(list(units_to_remove), reverse=True, key=get_model_unit_key(model))

    replaced_units: list[ReplacedUnit] = []

    for unit_to_remove in sorted_units:
        assert isinstance(unit_to_remove, ModelUnit), "Expected ModelUnit instance"
        assert unit_to_remove.block_idx >= 0, "Unit block index must be non-negative"
        assert unit_to_remove <= len(model.model.layers), (
            f"Unit block index {unit_to_remove.block_idx} out of bounds (max is {len(model.model.layers)})"
        )

        block = model.layers[unit_to_remove.block_idx]
        replacement_unit = ZeroBlock()

        replaced_unit = ReplacedUnit(
            block_idx=unit_to_remove.block_idx,
            unit_name=unit_to_remove.unit_name,
            replacement_unit=getattr(block, unit_to_remove.unit_name),
        )
        replaced_units.append(replaced_unit)

        setattr(block, unit_to_remove.unit_name, replacement_unit)

    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()

    def unprune():
        for replaced_unit in replaced_units:
            block = model.layers[replaced_unit.block_idx]
            setattr(block, replaced_unit.unit_name, replaced_unit.replacement_unit)

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

    return unprune


class ZeroBlock(nn.Module):
    def forward(self: Self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        return th.zeros_like(x)
