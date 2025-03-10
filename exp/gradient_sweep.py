import gc
from typing import Callable

import arguably
import asyncio
import torch as th

from core.model import MODELS
from exp.gradient_scaling import main as gradient_scaling_main
from exp.exp_data import get_exp_data, ExperimentConfig, GRADIENT_SCALING_OUT_SUBDIR


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
    out_dir: str,
    sub_dir: str,
    batchsize: int = 0,
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
            batchsize=batchsize,
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
            out_dir=out_dir,
            batchsize=batchsize,
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
    out_dir: str = "out",
    parallel: int = 1,
    batchsize: int = 0,
) -> None:
    completed_experiments = set(get_exp_data(out_dir, GRADIENT_SCALING_OUT_SUBDIR).keys())

    print("Experiment config:")
    print(f"    model_name: {model_name}")
    print(f"    dataset_name: {dataset_name}")
    print(f"    maxlen: {maxlen}")
    print(f"    dtype: {dtype}")
    print(f"    load_in_8bit: {load_in_8bit}")
    print(f"    load_in_4bit: {load_in_4bit}")


    world_size = parallel
    for rank in range(world_size):
        asyncio.run(
            sweep_task(
                gradient_scaling_main,
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
                out_dir,
                GRADIENT_SCALING_OUT_SUBDIR,
                batchsize=batchsize,
            )
        )

if __name__ == "__main__":
    arguably.run()
