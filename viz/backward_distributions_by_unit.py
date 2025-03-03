import os
import yaml
from dataclasses import dataclass

import arguably
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable

from core.model import MODELS
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR


FIGURES_DIR = "figures_backward"
COLORMAP = plt.get_cmap("viridis")


def figure_key(
    model_name: str,
    dataset_name: str,
    maxlen: int,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    n: int,
) -> str:
    return f"model={model_name},dataset={dataset_name},maxlen={maxlen},dtype={dtype},load_in_8bit={load_in_8bit},load_in_4bit={load_in_4bit},n={n},measure=mean"


@dataclass
class StatConfig:
    name: str
    title: str
    legend_name: str
    derived: None | Callable[[dict], th.Tensor] = None


STAT_CONFIGS = {
    "avg_cosine_similarity_by_depth": StatConfig(name="Cosine similarity", title="avg cosine similarity between nth-order gradient and true gradient by depth", legend_name="Depth {}"),
    "avg_cosine_similarity_by_unit": StatConfig(name="Cosine similarity", title="avg cosine similarity between nth-order gradient and true gradient by unit", legend_name="Unit {}"),
    "avg_l2_norm_by_depth": StatConfig(name="L2 norm", title="avg L2 norm of nth-order gradient by depth", legend_name="Depth {}"),
    "avg_l2_norm_by_unit": StatConfig(name="L2 norm", title="avg L2 norm of nth-order gradient by unit", legend_name="Unit {}"),
    "projected_onto_normalized_by_depth": StatConfig(name="Projection magnitude", title="avg magnitude of nth-order gradient projected onto true gradient by depth", derived=lambda row: row["avg_cosine_similarity_by_depth"] * row["avg_l2_norm_by_depth"] if row["avg_l2_norm_by_depth"] else None, legend_name="Depth {}"),
    "projected_onto_normalized_by_unit": StatConfig(name="Projection magnitude", title="avg magnitude of nth-order gradient projected onto true gradient by unit", derived=lambda row: row["avg_cosine_similarity_by_unit"] * row["avg_l2_norm_by_unit"] if row["avg_l2_norm_by_unit"] else None, legend_name="Unit {}"),
}
DERIVED_STAT_CONFIGS = {
    stat_name: stat_config
    for stat_name, stat_config in STAT_CONFIGS.items()
    if stat_config.derived is not None
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

    units_stats = {stat_name: None for stat_name in STAT_CONFIGS.keys()}  # stat, U, D, T
    exp_paths = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments)):
        exp_paths.append(exp_path)

        with open(os.path.join(exp_path, DATA_FILE), "r") as f:
            data = yaml.safe_load(f)["unit_stats"]
            data = sorted(data, key=lambda x: x["unit_index"])

        num_units = len(data)

        for unit_idx, row in enumerate(data):
            longest_stat = max(len(values) for values in row.values() if isinstance(values, list))

            # transpose
            unit_or_depth_rows = [{stat: None for stat in STAT_CONFIGS.keys()} for _ in range(longest_stat)]
            for stat, values in row.items():
                if stat not in STAT_CONFIGS.keys():
                    continue

                for unit_or_depth_idx, value in enumerate(values):
                    unit_or_depth_rows[unit_or_depth_idx][stat] = value

            # derived stats
            for derived_stat, config in DERIVED_STAT_CONFIGS.items():
                for unit_or_depth_idx, unit_or_depth_row in enumerate(unit_or_depth_rows):
                    unit_or_depth_row[derived_stat] = config.derived(unit_or_depth_row)

            for unit_or_depth_idx, unit_or_depth_row in enumerate(unit_or_depth_rows):
                for stat, value in unit_or_depth_row.items():
                    if units_stats[stat] is None:
                        units_stats[stat] = [None for _ in range(num_units)]

                    if units_stats[stat][unit_idx] is None:
                        units_stats[stat][unit_idx] = [[] for _ in range(len(unit_or_depth_rows))]

                    if value is None:
                        continue

                    units_stats[stat][unit_idx][unit_or_depth_idx].append(value)

    # units_stats is of shape stat, U, D, T
    # U is the unit, D is the depth or unit index (for units upstream of the gradient circuit), and T is the time (checkpoint) dimension

    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found"

    steps = [model_config.checkpoints[experiment[0].checkpoint_idx].step for experiment in sorted_experiments]

    for stat, config in STAT_CONFIGS.items():
        units_stat = units_stats[stat]

        for unit_idx, unit_or_depth_values in enumerate(units_stat):
            filtered_unit_or_depth_values = [(idx, values) for idx, values in enumerate(unit_or_depth_values) if len(values) > 0]

            colors = COLORMAP(th.linspace(0, 1, len(filtered_unit_or_depth_values)))

            for color_idx, (unit_or_depth_idx, values) in enumerate(filtered_unit_or_depth_values):
                plt.plot(steps, values, label=config.legend_name.format(unit_or_depth_idx), color=colors[color_idx])

            plt.xlabel("Checkpoint step")
            plt.ylabel(config.name)
            plt.legend()
            plt.title(config.title)

            key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit, n)
            image_dir = os.path.join(FIGURES_DIR, key, stat)
            os.makedirs(image_dir, exist_ok=True)
            plt.savefig(os.path.join(image_dir, f"{unit_idx}.png"))
            plt.clf()

if __name__ == "__main__":
    arguably.run()
