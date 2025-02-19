import os
import yaml
from math import isnan

import arguably
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal

from core.model import MODELS
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data


FIGURES_DIR = "figures"
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
NAN = float("nan")


def figure_key(
    model_name: str,
    dataset_name: str,
    maxlen: int,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    n: int,
    measure: Literal["mean", "median"],
) -> str:
    return f"model={model_name},dataset={dataset_name},maxlen={maxlen},dtype={dtype},load_in_8bit={load_in_8bit},load_in_4bit={load_in_4bit},n={n},measure={measure}"


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

    # TODO: on top of line graph also show bars indicating how many nans there are
    all_nth_order_relative_losses = [[] for _ in range(n)]
    exp_paths = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments)):
        exp_paths.append(exp_path)

        checkpoint_nth_order_relative_losses = [[] for _ in range(n)]
        with open(os.path.join(exp_path, DATA_FILE), "r") as f:
            data = yaml.safe_load(f)

            baseline = next(row for row in data if row["unit_indices"] == [])
            baseline_loss = baseline["subtractive_loss"]

            if not isnan(baseline_loss):
                for row in data:
                    subtractive_loss, unit_indices = row["subtractive_loss"], row["unit_indices"]
                    order = len(unit_indices) - 1

                    if isnan(subtractive_loss) or order < 0:
                        continue

                    checkpoint_nth_order_relative_losses[order].append(subtractive_loss - baseline_loss)

        checkpoint_nth_order_relative_losses = [th.tensor(order_losses[::-1]) for order_losses in checkpoint_nth_order_relative_losses]

        for order, relative_losses in enumerate(checkpoint_nth_order_relative_losses):
            all_nth_order_relative_losses[order].append(relative_losses)

    # all_nth_order_relative_losses is of shape O, T, N
    # O is the order, T is the time (checkpoint) dimension, and N is the number of circuit paths

    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found"

    steps = [model_config.checkpoints[experiment[0].checkpoint_idx].step for experiment in sorted_experiments]

    if measure == "mean":
        mean_losses = [[checkpoint_losses.mean().item() for checkpoint_losses in order_losses] for order_losses in all_nth_order_relative_losses]
        for order, mean_loss in enumerate(mean_losses):
            plt.plot(steps, mean_loss, label=f"Order {order + 1}", color=COLORS[order])
    elif measure == "median":
        bounds = th.tensor([0.25, 0.5, 0.75])
        nan_bounds = th.full_like(bounds, NAN)
        # O, T, N -> O, T, 3
        nth_order_bounds = [[th.quantile(checkpoint_losses, bounds) if checkpoint_losses.shape[0] > 0 else nan_bounds for checkpoint_losses in order_losses] for order_losses in all_nth_order_relative_losses]
        # O, T, 3 -> O, 3, T
        nth_order_bounds = [th.stack(order_bounds, dim=1) for order_bounds in nth_order_bounds]

        for order, bounds in enumerate(nth_order_bounds):
            plt.plot(steps, bounds[1], label=f"Order {order + 1}", color=COLORS[order])
            plt.fill_between(steps, bounds[0], bounds[2], alpha=0.2, color=COLORS[order])
    else:
        raise ValueError(f"Unknown measure {measure}")

    plt.xlabel("Checkpoint step")
    plt.ylabel("Relative Loss")
    plt.legend()

    key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit, n, measure)
    image_dir = os.path.join(FIGURES_DIR, key)
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, "figure.png"))

    # store a bunch of softlinks to the data that was used to generate the figure
    for exp_path in exp_paths:
        symlink_path = os.path.abspath(os.path.join(image_dir, os.path.basename(exp_path)))

        if os.path.exists(symlink_path):
            continue

        os.symlink(os.path.abspath(exp_path), symlink_path)

    plt.show()

if __name__ == "__main__":
    arguably.run()
