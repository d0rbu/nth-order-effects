import os
import yaml
import time

import arguably
import torch as th
from tqdm import tqdm

from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.gradients import compute_gradients
from exp.exp_data import DTYPE_MAP, DATA_FILE, METADATA_FILE, GRADIENT_SCALING_OUT_SUBDIR

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
    out_dir: str = "out",
    batchsize: int = 0,
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer, checkpoint = get_model_and_tokenizer(model_name, checkpoint_idx, model_kwargs=model_kwargs)

    gradients, attention_mask = compute_gradients(model, checkpoint, tokenizer, dataset, max_token_length=maxlen, batchsize=batchsize)
    gradient_scaling_factors = []  # U, N
    for input_gradient, output_gradient in tqdm(zip(gradients[:-1], gradients[1:]), desc="Computing gradient scaling factors", total=len(gradients) - 1, leave=False):
        unit_gradient = input_gradient - output_gradient  # B, T, D
        unit_gradient_norm = th.linalg.norm(unit_gradient, dim=-1)  # B, T
        output_gradient_norm = th.linalg.norm(output_gradient, dim=-1)  # B, T

        unit_gradient_scaling_factors = unit_gradient_norm / output_gradient_norm  # B, T
        gradient_scaling_factors.append(unit_gradient_scaling_factors[attention_mask].tolist())

    final_data = gradient_scaling_factors

    out_timestamp_dir = str(int(time.time() * 1000))
    final_out_dir = os.path.join(out_dir, GRADIENT_SCALING_OUT_SUBDIR, out_timestamp_dir)

    out_filepath = os.path.join(final_out_dir, DATA_FILE)
    metadata_out_filepath = os.path.join(final_out_dir, METADATA_FILE)

    os.makedirs(final_out_dir, exist_ok=True)
    with open(out_filepath, "w") as f:
        yaml.dump(final_data, f)

    with open(metadata_out_filepath, "w") as f:
        metadata = {
            "model": model_name,
            "dataset": dataset_name,
            "checkpoint_idx": checkpoint_idx,
            "checkpoint_metadata": {
                "step": checkpoint.step,
                "num_tokens": checkpoint.num_tokens,
                "model_config": {
                    "hf_name": checkpoint.model_config.hf_name,
                    "surgical_class": checkpoint.model_config.surgical_class.__name__,
                }
            } if checkpoint is not None else None,
            "maxlen": maxlen,
            "device": device,
            "dtype": dtype,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "out_dir": out_dir,
            "batchsize": batchsize,
        }

        yaml.dump(metadata, f)


if __name__ == "__main__":
    command = arguably.command(main)

    arguably.run()
