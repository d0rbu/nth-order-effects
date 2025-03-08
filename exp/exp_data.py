import os
import yaml
from dataclasses import dataclass

import torch as th


DTYPE_MAP = {
    "bf16": th.bfloat16,
    "fp32": th.float32,
    "fp16": th.float16,
}
DATA_FILE = "data.yaml"
METADATA_FILE = "metadata.yaml"
CONTRIBUTIONS_OUT_SUBDIR = "contributions"
BACKWARD_CONTRIBUTIONS_OUT_SUBDIR = "backward_contributions"


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
    batchsize: int = 0,
    num_samples: int = 0,
    seed: int = 44,
    sample_by_circuit: bool = False,


def get_exp_data(
    out_dir: str = "out",
    out_subdir: str = CONTRIBUTIONS_OUT_SUBDIR,
) -> dict[ExperimentConfig, str]:
    exp_dir = os.path.join(out_dir, out_subdir)
    os.makedirs(exp_dir, exist_ok=True)

    completed_experiments: dict[ExperimentConfig, str] = {}
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

            completed_experiment = ExperimentConfig(
                model_name=metadata["model"],
                dataset_name=metadata["dataset"],
                checkpoint_idx=metadata["checkpoint_idx"],
                maxlen=metadata["maxlen"],
                device=metadata["device"],
                dtype=metadata["dtype"],
                load_in_8bit=metadata["load_in_8bit"],
                load_in_4bit=metadata["load_in_4bit"],
                n=metadata["n"],
                batchsize=metadata["batchsize"],
                num_samples=metadata["num_samples"],
                seed=metadata["seed"],
                sample_by_circuit=metadata["sample_by_circuit"],
            )

            completed_experiments[completed_experiment] = exp_path

    return completed_experiments
