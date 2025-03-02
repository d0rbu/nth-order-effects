import os
import yaml
from math import isnan
from dataclasses import dataclass

import arguably
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal

from core.model import MODELS
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR


FIGURES_DIR = "figures_backward"
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


@dataclass
class StatConfig:
    name: str
    title: str


STAT_CONFIGS = {
    "cosine_similarity": StatConfig(name="Cosine similarity", title="avg cosine similarity between nth-order gradient and true gradient"),
}


@arguably.command
def main(
    *args,
    model_name: str = "pythia410m",
    dataset_name: str = "redpajama-1",
    maxlen: int = 256,
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 3,
    out_dir: str = "out",
    measure: str = "mean",
) -> None:
    completed_experiments = get_exp_data(out_dir, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR)
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

    all_stat_distributions = {stat_name: [[] for _ in range(n)] for stat_name in STAT_CONFIGS.keys()}  # stat, O, T, N
    exp_paths = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments)):
        exp_paths.append(exp_path)

        with open(os.path.join(exp_path, DATA_FILE), "r") as f:
            data = yaml.safe_load(f)["all_stats"]

        for stat, config in STAT_CONFIGS.items():
            checkpoint_nth_order_distribution = [[] for _ in range(n)]  # O, N

            for row in data:
                stat_value = row.get(stat, None)
                order = len(row.get("unit_indices", [])) - 1

                if stat_value is None or order < 0:
                    continue

                checkpoint_nth_order_distribution[order].append(stat_value)

            for order, values in enumerate(checkpoint_nth_order_distribution):
                all_stat_distributions[stat][order].append(th.tensor(values))

    # all_stat_distributions is of shape stat, O, T, N
    # O is the order, T is the time (checkpoint) dimension, and N is the number of circuit paths

    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found"

    steps = [model_config.checkpoints[experiment[0].checkpoint_idx].step for experiment in sorted_experiments]

    for stat, config in STAT_CONFIGS.items():
        stat_distribution = all_stat_distributions[stat]

        if measure == "mean":
            mean_values = [[checkpoint_values.mean().item() for checkpoint_values in order_values] for order_values in stat_distribution]
            for order, mean_value in enumerate(mean_values):
                plt.plot(steps, mean_value, label=f"Order {order + 1}", color=COLORS[order])
        elif measure == "median":
            bounds = th.tensor([0.25, 0.5, 0.75])
            nan_bounds = th.full_like(bounds, NAN)
            # O, T, N -> O, T, 3
            nth_order_bounds = [[th.quantile(checkpoint_values, bounds) if checkpoint_values.shape[0] > 0 else nan_bounds for checkpoint_values in order_values] for order_values in stat_distribution]
            # O, T, 3 -> O, 3, T
            nth_order_bounds = [th.stack(order_bounds, dim=1) for order_bounds in nth_order_bounds]

            for order, bounds in enumerate(nth_order_bounds):
                plt.plot(steps, bounds[1], label=f"Order {order + 1}", color=COLORS[order])
                plt.fill_between(steps, bounds[0], bounds[2], alpha=0.2, color=COLORS[order])
        else:
            raise ValueError(f"Unknown measure {measure}")

        plt.xlabel("Checkpoint step")
        plt.ylabel(config.name)
        plt.legend()
        plt.title(config.title)

        key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit, n, measure)
        image_dir = os.path.join(FIGURES_DIR, key)
        os.makedirs(image_dir, exist_ok=True)
        plt.savefig(os.path.join(image_dir, f"{stat}.png"))

    # # store a bunch of softlinks to the data that was used to generate the figure
    # for exp_path in exp_paths:
    #     symlink_path = os.path.abspath(os.path.join(image_dir, os.path.basename(exp_path)))

    #     if os.path.exists(symlink_path):
    #         continue

    #     os.symlink(os.path.abspath(exp_path), symlink_path)

if __name__ == "__main__":
    arguably.run()
