from dataclasses import dataclass, fields, field

import torch as th


class ActivationMaskMixin:
    @staticmethod
    def get_mask_subset(mask: list[str], prefix: list[str]) -> list[str]:
        mask_subset = []  # subset of the mask which corresponds to this element, with the prefix removed

        for mask_element in mask:
            mask_element_path = mask_element.split(".")

            for path_idx, (mask_part, prefix_part) in enumerate(zip(mask_element_path, prefix)):
                if mask_part != "*" and mask_part != prefix_part:
                    break
            else:
                mask_element_path_without_prefix = mask_element_path[path_idx + 1:]
                mask_element_without_prefix = ".".join(mask_element_path_without_prefix)
                mask_subset.append(mask_element_without_prefix)

        return mask_subset

    def apply_activation_mask(self, mask: bool | list[str], called_recursively: bool = False) -> None:
        if mask is True:
            return

        for child in fields(self):
            field_value = getattr(self, child.name, None)

            if field_value is None:
                continue

            if isinstance(field_value, ActivationMaskMixin):
                tree_elements = [((child.name,), field_value)]
                leaf_elements = []
            elif isinstance(field_value, list):
                tree_elements = [((child.name, str(idx)), element) for idx, element in enumerate(field_value) if isinstance(element, ActivationMaskMixin)]
                leaf_elements = [((child.name, str(idx)), element) for idx, element in enumerate(field_value) if not isinstance(element, ActivationMaskMixin)]
            elif isinstance(field_value, dict):
                tree_elements = [((child.name, str(element_key)), element_value) for element_key, element_value in field_value.items() if isinstance(element_value, ActivationMaskMixin)]
                leaf_elements = [((child.name, str(element_key)), element_value) for element_key, element_value in field_value.items() if not isinstance(element_value, ActivationMaskMixin)]
            else:
                tree_elements = []
                leaf_elements = [((child.name,), field_value)]

            for element_path, element in tree_elements:
                if isinstance(mask, bool):
                    new_mask = mask
                else:
                    new_mask = self.get_mask_subset(mask, element_path)

                element.apply_activation_mask(new_mask, called_recursively=True)

            for element_path, element in leaf_elements:
                if element_path[-1] in ("output", "loss") and not called_recursively:
                    mask_value = True
                elif isinstance(mask, bool):
                    mask_value = mask
                else:
                    new_mask = self.get_mask_subset(mask, element_path)
                    mask_value = len(new_mask) > 0

                current_parent = self
                current_element = element
                
                for element_part in element_path[1:]:
                    current_parent = current_element
                    current_element = current_parent[element_part]

                if current_parent is self:
                    setattr(current_parent, element_path[-1], current_element if mask_value else None)
                else:
                    current_parent[element_path[-1]] = current_element if mask_value else None

def add_activation_masks(activation_mask_0: bool | list[str], activation_mask_1: bool | list[str]) -> bool | list[str]:
    if activation_mask_0 is True or activation_mask_1 is True:
        return True
    elif activation_mask_0 is False and activation_mask_1 is False:
        return False
    else:
        return list(set(activation_mask_0) | set(activation_mask_1))

@dataclass
class AttentionActivations(ActivationMaskMixin):
    query_activation: th.FloatTensor | None = None
    key_activation: th.FloatTensor | None = None
    value_activation: th.FloatTensor | None = None
    normed_query_activation: th.FloatTensor | None = None
    normed_key_activation: th.FloatTensor | None = None
    rotated_query_activation: th.FloatTensor | None = None
    rotated_key_activation: th.FloatTensor | None = None
    attention_map: th.FloatTensor | None = None
    attention_output: th.FloatTensor | None = None
    output: th.FloatTensor | None = None  # o_proj(attention_output)

@dataclass
class MLPActivations(ActivationMaskMixin):
    gate_proj_activation: th.FloatTensor | None = None
    gate_proj_nonlinear_activation: th.FloatTensor | None = None
    up_proj_activation: th.FloatTensor | None = None
    hidden_activation: th.FloatTensor | None = None
    output: th.FloatTensor | None = None  # down_proj(hidden_activation)

@dataclass
class DecoderLayerActivations(ActivationMaskMixin):
    attention_normed_input: th.FloatTensor | None = None  # norm(input)
    attention_activations: AttentionActivations = field(default_factory=AttentionActivations)
    attention_dropped_output: th.FloatTensor | None = None  # dropped(attention_activations.output)
    attention_normed_output: th.FloatTensor | None = None  # norm(attention_activations.output)
    attention_output: th.FloatTensor | None = None  # norm(attention_activations.output) + residual
    mlp_normed_input: th.FloatTensor | None = None
    mlp_activations: MLPActivations = field(default_factory=MLPActivations)
    mlp_dropped_output: th.FloatTensor | None = None  # dropped(mlp_activations.output)
    mlp_normed_output: th.FloatTensor | None = None  # norm(mlp_activations.output)
    output: th.FloatTensor | None = None  # norm(mlp_activations.output) + residual

@dataclass
class ModelActivations(ActivationMaskMixin):
    residual_base: th.FloatTensor | None = None
    layer_activations: list[DecoderLayerActivations] = field(default_factory=list)
    output: th.FloatTensor | None = None  # norm(final residual state)

@dataclass
class CausalLMActivations(ActivationMaskMixin):
    model_activations: ModelActivations = field(default_factory=ModelActivations)
    logits: th.FloatTensor | None = None
    loss: th.FloatTensor | None = None

ActivationClass = CausalLMActivations | ModelActivations | DecoderLayerActivations | AttentionActivations | MLPActivations
