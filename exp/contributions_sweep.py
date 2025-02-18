from dataclasses import dataclass
import os
import gc
import yaml

import arguably
import torch as th

from core.model import MODELS
from exp.contributions import main as subtract_contributions_main
from exp.contributions import DATA_FILE, METADATA_FILE, OUT_SUBDIR


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    dataset_name: str
    checkpoint_idx: int | None
    maxlen: int
    device: str
    dtype: str
    load_in_8bit: bool
    load_in_4bit: bool
    n: int

@arguably.command
def sweep(
    *args,
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-1",
    maxlen: int = 128,
    device: str = "cuda",
    dtype: str = "bf16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    n: int = 3,
    out_dir: str = "out",
) -> None:
    max_checkpoint_idx = len(MODELS[model_name].checkpoints) - 1

    exp_dir = os.path.join(out_dir, OUT_SUBDIR)
    os.makedirs(exp_dir, exist_ok=True)

    completed_experiments: set[ExperimentConfig] = set()
    for subdir in os.listdir(exp_dir):
        exp_path = os.path.join(exp_dir, subdir)
        if not os.path.isdir(exp_path):
            continue

        # check if the experiment is already completed
        data_path = os.path.join(exp_path, DATA_FILE)
        metadata_path = os.path.join(exp_path, METADATA_FILE)
        if not os.path.exists(data_path) or not os.path.exists(metadata_path):
            continue

        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
            if not metadata:
                continue

            completed_experiments.add(
                ExperimentConfig(
                    model_name=metadata["model"],
                    dataset_name=metadata["dataset"],
                    checkpoint_idx=metadata["checkpoint_idx"],
                    maxlen=metadata["maxlen"],
                    device=metadata["device"],
                    dtype=metadata["dtype"],
                    load_in_8bit=metadata["load_in_8bit"],
                    load_in_4bit=metadata["load_in_4bit"],
                    n=metadata["n"],
                )
            )

    # fill in checkpoint_idx. so we start at 0, then max_checkpoint_idx, then max_checkpoint_idx // 2, then max_checkpoint_idx // 4, then 3 * max_checkpoint_idx // 4, etc.
    checkpoint_indices = [max_checkpoint_idx]
    num_divisions = 1
    while True:
        if len(checkpoint_indices) <= 0:
            num_divisions *= 2
            for i in range(num_divisions):
                checkpoint_indices.append(max_checkpoint_idx * i // num_divisions)

        checkpoint_idx = checkpoint_indices.pop(0)
        experiment = ExperimentConfig(
            model_name=model_name,
            dataset_name=dataset_name,
            checkpoint_idx=checkpoint_idx,
            maxlen=maxlen,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            n=n,
        )

        if experiment in completed_experiments:
            continue

        subtract_contributions_main(
            model_name=model_name,
            dataset_name=dataset_name,
            checkpoint_idx=checkpoint_idx,
            maxlen=maxlen,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            n=n,
            out_dir=out_dir,
        )
        th.cuda.empty_cache()
        gc.collect()

        completed_experiments.add(experiment)

if __name__ == "__main__":
    arguably.run()
