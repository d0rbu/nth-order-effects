import gc
from typing import Callable
from dataclasses import dataclass

import arguably
import asyncio
import torch as th

from core.model import MODELS
from exp.contributions import main as subtract_contributions_main
from exp.contributions_backward import main as backward_contributions_main
from exp.exp_data import get_exp_data, ExperimentConfig, CONTRIBUTIONS_OUT_SUBDIR, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR


@dataclass
class ExperimentType:
    main: Callable
    sub_dir: str

EXPERIMENT_TYPES = {
    "contributions": ExperimentType(main=subtract_contributions_main, sub_dir=CONTRIBUTIONS_OUT_SUBDIR),
    "contributions_backward": ExperimentType(main=backward_contributions_main, sub_dir=BACKWARD_CONTRIBUTIONS_OUT_SUBDIR),
}


async def sweep_task(
    main: Callable,
    completed_experiments: set[ExperimentConfig],
    rank: int,
    world_size: int,
    model_name: str,
    dataset_name: str,
    maxlen: int,
    device: str,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    n: int,
    out_dir: str,
    sub_dir: str,
) -> None:
    num_checkpoints = len(MODELS[model_name].checkpoints)
    max_checkpoint_idx = num_checkpoints - 1

    # fill in checkpoint_idx. so we start at 0, then max_checkpoint_idx, then max_checkpoint_idx // 2, then max_checkpoint_idx // 4, then 3 * max_checkpoint_idx // 4, etc.
    checkpoint_indices = [max_checkpoint_idx]
    num_divisions = 1
    for exp_idx in range(num_checkpoints):
        if len(checkpoint_indices) <= 0:
            num_divisions *= 2
            for i in range(num_divisions):
                checkpoint_indices.append(max_checkpoint_idx * i // num_divisions)

        checkpoint_idx = checkpoint_indices.pop(0)

        if exp_idx % world_size != rank:
            continue

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

        main(
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
    exp_type: str = "contributions",
    parallel: int = 1,
) -> None:
    if (experiment_type := EXPERIMENT_TYPES.get(exp_type, None)) is None:
        raise ValueError(f"Invalid exp_type: {exp_type}")

    completed_experiments = set(get_exp_data(out_dir, experiment_type.sub_dir).keys())

    print("Experiment config:")
    print(f"    model_name: {model_name}")
    print(f"    dataset_name: {dataset_name}")
    print(f"    maxlen: {maxlen}")
    print(f"    dtype: {dtype}")
    print(f"    load_in_8bit: {load_in_8bit}")
    print(f"    load_in_4bit: {load_in_4bit}")
    print(f"    n: {n}")


    world_size = parallel
    for rank in range(world_size):
        asyncio.run(
            sweep_task(
                experiment_type.main,
                completed_experiments,
                rank,
                world_size,
                model_name,
                dataset_name,
                maxlen,
                device,
                dtype,
                load_in_8bit,
                load_in_4bit,
                n,
                out_dir,
                experiment_type.sub_dir,
            )
        )

if __name__ == "__main__":
    arguably.run()
