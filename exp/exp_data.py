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
GRADIENT_SCALING_OUT_SUBDIR = "gradient_scaling"


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    dataset_name: str
    checkpoint_idx: int | None = None
    maxlen: int = 2048
    device: str = "cuda"
    dtype: str = "fp32"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    n: int = 0
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
                checkpoint_idx=metadata.get("checkpoint_idx", None),
                maxlen=metadata.get("maxlen", 2048),
                device=metadata.get("device", "cuda"),
                dtype=metadata.get("dtype", "fp32"),
                load_in_8bit=metadata.get("load_in_8bit", False),
                load_in_4bit=metadata.get("load_in_4bit", False),
                n=metadata.get("n", 0),
                batchsize=metadata.get("batchsize", 0),
                num_samples=metadata.get("num_samples", 0),
                seed=metadata.get("seed", 44),
                sample_by_circuit=metadata.get("sample_by_circuit", False),
            )

            completed_experiments[completed_experiment] = exp_path

    return completed_experiments
