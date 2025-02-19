import os
import yaml

import arguably
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal

from core.model import MODELS
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data


COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


@arguably.command
def main(
    *args,
    model_name: str = "pythia410m",
    dataset_name: str = "redpajama-1",
    maxlen: int = 256,
    dtype: str = "fp16",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 3,
    out_dir: str = "out",
    measure: str = "mean",
) -> None:
    completed_experiments = get_exp_data(out_dir)
    filtered_experiments = {
        completed_experiment: exp_path
        for completed_experiment, exp_path in completed_experiments.items()
        if completed_experiment.model_name == model_name
        and completed_experiment.dataset_name == dataset_name
        and completed_experiment.dtype == dtype
        and completed_experiment.load_in_8bit == load_in_8bit
        and completed_experiment.load_in_4bit == load_in_4bit
        and completed_experiment.maxlen == maxlen
        and completed_experiment.n == n
        and completed_experiment.checkpoint_idx is not None
    }
    assert len(filtered_experiments) > 0, "No experiments found with the given parameters"
    sorted_experiments = sorted(filtered_experiments.items(), key=lambda x: x[0].checkpoint_idx)

    all_nth_order_relative_losses = [[] for _ in range(n)]
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments)):
        current_nth_order_relative_losses = [[] for _ in range(n)]
        with open(os.path.join(exp_path, DATA_FILE), "r") as f:
            data = yaml.safe_load(f)

            baseline = next(row for row in data if row["unit_indices"] == [])
            baseline_loss = baseline["subtractive_loss"]

            for row in data:
                subtractive_loss, unit_indices = row["subtractive_loss"], row["unit_indices"]
                order = len(unit_indices) - 1

                current_nth_order_relative_losses[order].append(subtractive_loss - baseline_loss)

        current_nth_order_relative_losses = [th.tensor(losses[::-1]) for losses in current_nth_order_relative_losses]

        for order, relative_losses in enumerate(current_nth_order_relative_losses):
            all_nth_order_relative_losses[order].append(relative_losses)

    all_nth_order_relative_losses = [th.stack(losses) for losses in all_nth_order_relative_losses]

    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found"

    steps = [model_config.checkpoints[experiment[0].checkpoint_idx].step for experiment in sorted_experiments]

    if measure == "mean":
        mean_losses = [losses.mean(dim=-1) for losses in all_nth_order_relative_losses]
        for order, mean_loss in enumerate(mean_losses):
            plt.plot(steps, mean_loss, label=f"Order {order + 1}", color=COLORS[order])
    elif measure == "median":
        bounds = th.tensor([0.25, 0.5, 0.75])
        nth_order_bounds = [th.quantile(losses, bounds, dim=-1) for losses in all_nth_order_relative_losses]

        for order, bounds in enumerate(nth_order_bounds):
            plt.plot(steps, bounds[1], label=f"Order {order + 1}", color=COLORS[order])
            plt.fill_between(steps, bounds[0], bounds[2], alpha=0.2, color=COLORS[order])
    else:
        raise ValueError(f"Unknown measure {measure}")

    plt.xlabel("Checkpoint step")
    plt.ylabel("Relative Loss")

    plt.legend()
    plt.show()

    input("Press Enter to continue...")

if __name__ == "__main__":
    arguably.run()
