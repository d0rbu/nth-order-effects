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

from core.data import get_dataset
from core.model import get_model_and_tokenizer, MODELS
from core.nth_order import compute_nth_order_deltas, NthOrderDelta


@dataclass
class DeltaLoss:
    unit_indices: tuple[int]
    subtractive_loss: float

DTYPE_MAP = {
    "bf16": th.bfloat16,
    "fp32": th.float32,
    "fp16": th.float16,
}
DATA_FILE = "data.yaml"
METADATA_FILE = "metadata.yaml"
OUT_SUBDIR = __file__.split("/")[-1].replace(".py", "")

@arguably.command
def main(
    *args,
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    maxlen: int = 512,
    device: str = "cuda",
    dtype: str = "bf16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 2,
    out_dir: str = "out",
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer, model_config = get_model_and_tokenizer(model_name, checkpoint_idx, model_kwargs=model_kwargs)

    deltas, depth_deltas, units_deltas, final_state, inputs = compute_nth_order_deltas(model, tokenizer, dataset, stop_n=n, max_token_length=maxlen)

    loss_fn = partial(model.loss_function, labels=inputs["labels"], vocab_size=model.config.vocab_size)

    output_norm = model.model.norm
    lm_head = model.get_output_embeddings()
    output_module = nn.Sequential(output_norm, lm_head)

    with th.no_grad():
        delta_losses = compute_losses(final_state, deltas, loss_fn, output_module)

    del dataset, model_kwargs, model, tokenizer, deltas, depth_deltas, units_deltas, final_state, inputs, loss_fn, output_norm, lm_head, output_module
    th.cuda.empty_cache()
    gc.collect()

    sorted_delta_losses = sorted(delta_losses, key=lambda x: x.subtractive_loss, reverse=True)
    final_data = [asdict(x) for x in sorted_delta_losses]

    out_timestamp_dir = str(int(time.time()))
    final_out_dir = os.path.join(out_dir, OUT_SUBDIR, out_timestamp_dir)

    out_filepath = os.path.join(final_out_dir, DATA_FILE)
    metadata_out_filepath = os.path.join(final_out_dir, METADATA_FILE)

    os.makedirs(final_out_dir, exist_ok=True)
    with open(out_filepath, "w") as f:
        yaml.dump(final_data, f)

    with open(metadata_out_filepath, "w") as f:
        checkpoint = model_config.checkpoints[checkpoint_idx] if checkpoint_idx is not None else None

        metadata = {
            "dataset": dataset_name,
            "checkpoint_idx": checkpoint_idx,
            "checkpoint_metadata": asdict(checkpoint) if checkpoint is not None else None,
            "maxlen": maxlen,
            "device": device,
            "dtype": dtype,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "n": n,
            "out_dir": out_dir,
        }

        yaml.dump(metadata, f)

def compute_losses(
    final_state: th.Tensor,
    root: NthOrderDelta,
    loss_fn: Callable[[th.Tensor], th.Tensor],
    output_module: nn.Module,
) -> list[DeltaLoss]:
    delta_loss = loss_fn(output_module(final_state - root.delta.to(final_state.device))).item()

    losses = [DeltaLoss(root.unit_indices(), delta_loss)]
    for child in root.children:
        losses.extend(compute_losses(final_state, child, loss_fn, output_module))

    return losses


if __name__ == "__main__":
    arguably.run()
