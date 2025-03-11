import os
import re
import yaml
from dataclasses import dataclass

import arguably
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Literal, Callable

from core.model import MODELS
from core.nth_order import calculate_max_by_order
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data, GRADIENT_SCALING_OUT_SUBDIR


FIGURES_DIR = "figures_gradient_scaling"
COLORMAP = plt.get_cmap("viridis")
ALL_MEASURES = ["mean", "median", "bounds"]


def figure_key(
    model_name: str,
    dataset_name: str,
    maxlen: int,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
) -> str:
    return f"model={model_name},dataset={dataset_name},maxlen={maxlen},dtype={dtype},load_in_8bit={load_in_8bit},load_in_4bit={load_in_4bit}"


@arguably.command
def main(
    *args,
    model_name: str = "pythia410m",
    dataset_name: str = "redpajama-1",
    maxlen: int = 256,
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    out_dir: str = "out",
    measure: str = "all",
) -> None:
    take_all_measures = measure == "all"
    measures = ALL_MEASURES if take_all_measures else [measure]

    completed_experiments = get_exp_data(out_dir, GRADIENT_SCALING_OUT_SUBDIR)
    filtered_experiments = {
        completed_experiment: exp_path
        for completed_experiment, exp_path in completed_experiments.items()
        if completed_experiment.model_name == model_name
        and completed_experiment.dataset_name == dataset_name
        and completed_experiment.dtype == dtype
        and completed_experiment.load_in_8bit == load_in_8bit
        and completed_experiment.load_in_4bit == load_in_4bit
        and completed_experiment.maxlen == maxlen
        and completed_experiment.checkpoint_idx is not None
    }
    assert len(filtered_experiments) > 0, "No experiments found with the given parameters"
    sorted_experiments = sorted(filtered_experiments.items(), key=lambda x: x[0].checkpoint_idx)

    all_distributions = []  # U, T, N
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments)):
        with open(os.path.join(exp_path, DATA_FILE), "r") as f:
            raw_data = yaml.safe_load(f)
            data = raw_data  # U, N

        for unit_idx, unit_data in enumerate(data):
            if len(all_distributions) <= unit_idx:
                all_distributions.append([])
            all_distributions[unit_idx].append(unit_data)
    all_distributions = th.tensor(all_distributions)  # U, T, N

    num_units, num_timesteps, _ = all_distributions.shape

    nan_mask = th.isnan(all_distributions)  # to clear the pesky nans from loss being shifted by 1
    all_distributions = all_distributions[~nan_mask].view(num_units, num_timesteps, -1)

    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found"

    steps = [model_config.checkpoints[experiment[0].checkpoint_idx].step for experiment in sorted_experiments]
    colors = COLORMAP(th.linspace(0, 1, all_distributions.shape[0]))

    for measure in measures:
        if measure == "mean":
            mean_values = all_distributions.mean(dim=-1)  # U, T
            for unit_idx, mean_value in enumerate(mean_values):
                plt.plot(steps, mean_value, label=f"Unit {unit_idx}", color=colors[unit_idx])
        elif measure == "median":
            bounds = th.tensor([0.25, 0.5, 0.75])
            # U, T, N -> 3, U, T
            unit_bounds = th.quantile(all_distributions, bounds, dim=-1)
            # 3, U, T -> U, 3, T
            unit_bounds = unit_bounds.permute(1, 0, 2)
            for unit_idx, timeseries_bounds in enumerate(unit_bounds):
                plt.plot(steps, timeseries_bounds[1], label=f"Unit {unit_idx}", color=colors[unit_idx])
                plt.fill_between(steps, timeseries_bounds[0], timeseries_bounds[2], alpha=0.2, color=colors[unit_idx])
        elif measure == "bounds":
            bounds = th.tensor([0.0, 0.5, 1.0])
            # U, T, N -> 3, U, T
            unit_bounds = th.quantile(all_distributions, bounds, dim=-1)
            # 3, U, T -> U, 3, T
            unit_bounds = unit_bounds.permute(1, 0, 2)
            for unit_idx, timeseries_bounds in enumerate(unit_bounds):
                plt.plot(steps, timeseries_bounds[1], label=f"Unit {unit_idx}", color=colors[unit_idx])
                plt.fill_between(steps, timeseries_bounds[0], timeseries_bounds[2], alpha=0.2, color=colors[unit_idx])
        else:
            raise ValueError(f"Unknown measure {measure}")

        plt.xlabel("Checkpoint step")
        plt.ylabel(f"Scaling factor {measure}")
        plt.legend()
        plt.title("Gradient scaling factor by unit")

        key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit)
        image_dir = os.path.join(FIGURES_DIR, key)
        os.makedirs(image_dir, exist_ok=True)

        # make the plot 4096x4096 pixels
        fig = plt.gcf()
        fig.set_size_inches(4096 / fig.dpi, 4096 / fig.dpi)

        if take_all_measures:
            plt.savefig(os.path.join(image_dir, f"{measure}.png"))
        else:
            plt.show()

        plt.clf()

if __name__ == "__main__":
    arguably.run()
