import torch as th
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from more_itertools import batched

from core.model import Checkpoint, SurgicalModel
from core.surgical_gpt_neox import SurgicalGPTNeoXForCausalLM
from core.surgical_olmo import SurgicalOlmo2ForCausalLM


def compute_gradients(
    model: SurgicalModel,
    checkpoint: Checkpoint | None,
    tokenizer: PreTrainedTokenizerBase,
    dataset: list[str],
    max_token_length: int = 512,
    batchsize: int = 0,
) -> tuple[list[th.Tensor], th.Tensor]:
    """Compute the gradients at each layer of the model for the given dataset."""

    if batchsize == 0:
        batchsize = len(dataset)

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

    gradients = [None] * (num_units + 1)

    # batch each value in the inputs dictionary
    for batch_indices in tqdm(batched(range(len(dataset)), batchsize), desc="Computing batches", leave=False, total=len(dataset) // batchsize):
        batch_indices_tensor = th.tensor(batch_indices)
        batch = {key: value[batch_indices_tensor] for key, value in inputs.items()}

        match model:
            case SurgicalGPTNeoXForCausalLM():
                activations = model(
                    **batch,
                    activation_mask=["model_activations.layer_activations.*.output", "loss", "model_activations.residual_base"],
                )
                inputs_embeds = activations.model_activations.residual_base
                unit_activations = [inputs_embeds] + [layer_activation.output for layer_activation in activations.model_activations.layer_activations]
            case SurgicalOlmo2ForCausalLM():
                activations = model(
                    **batch,
                    activation_mask=[
                        "model_activations.layer_activations.*.output",
                        "model_activations.layer_activations.*.attention_normed_output",
                        "model_activations.layer_activations.*.mlp_normed_output",
                        "loss",
                        "model_activations.residual_base"
                    ],
                )
                inputs_embeds = activations.model_activations.residual_base
                layer_activations = [
                    [layer_activation.attention_output, layer_activation.output]
                    for layer_activation in activations.model_activations.layer_activations
                ]
                # flatten so we have alternating attention output, mlp output, attention output, mlp output, ...
                unit_activations = [inputs_embeds] + [activation for layer_activation in layer_activations for activation in layer_activation]

        current_gradients = [
            th.autograd.grad(
                activations.loss,
                unit_activation,
                retain_graph=True,
            )[0].cpu()[:, :-1]  # the last gradient is nothing because to compute the loss we shift the labels by 1
            for unit_activation in unit_activations
        ]

        for unit_idx, gradient in enumerate(current_gradients):
            if gradients[unit_idx] is None:
                gradients[unit_idx] = gradient
            else:
                gradients[unit_idx] = th.cat([gradients[unit_idx], gradient], dim=0)

    return gradients, inputs["attention_mask"]
