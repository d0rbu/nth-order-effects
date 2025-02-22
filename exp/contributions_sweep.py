import gc

import arguably
import torch as th

from core.model import MODELS
from exp.contributions import main as subtract_contributions_main
from exp.exp_data import get_exp_data, ExperimentConfig

@arguably.command
def sweep(
    *args,
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-1",
    maxlen: int = 128,
    device: str = "cuda",
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 3,
    out_dir: str = "out",
) -> None:
    max_checkpoint_idx = len(MODELS[model_name].checkpoints) - 1

    completed_experiments = set(get_exp_data(out_dir).keys())

    print("Experiment config:")
    print(f"    model_name: {model_name}")
    print(f"    dataset_name: {dataset_name}")
    print(f"    maxlen: {maxlen}")
    print(f"    dtype: {dtype}")
    print(f"    load_in_8bit: {load_in_8bit}")
    print(f"    load_in_4bit: {load_in_4bit}")
    print(f"    n: {n}")

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

        print(f"Running experiment with checkpoint_idx={checkpoint_idx}")

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
