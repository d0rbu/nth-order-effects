import os
import yaml
import arguably
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp.exp_data import get_exp_data, PRUNED_OUT_SUBDIR, DATA_FILE_YAML
from core.model import MODELS

FIGURES_PRUNED_DIR = "figures_pruned"
COLORMAP = plt.get_cmap("viridis")

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
    model_name: str = "olmo2-7b",
    dataset_name: str = "redpajama-1",
    maxlen: int = 512,
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    out_dir: str = "out",
    skip_first: int = 0,
    cut_edges: int = 1,
):
    completed_experiments = get_exp_data(out_dir, PRUNED_OUT_SUBDIR)
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

    key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit)
    figures_dir = os.path.join(FIGURES_PRUNED_DIR, key)
    os.makedirs(figures_dir, exist_ok=True)

    # Load all pruned results for all checkpoints
    all_pruned_models = []  # list of [list of dicts] for each checkpoint
    checkpoint_steps = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments), leave=False):
        data_path = os.path.join(exp_path, DATA_FILE_YAML)
        with open(data_path, "r") as f:
            pruned_models = yaml.safe_load(f)
        all_pruned_models.append(pruned_models)
        checkpoint_steps.append(experiment.checkpoint_step)

    # Optionally skip the first n datapoints
    if skip_first > 0:
        checkpoint_steps = checkpoint_steps[skip_first:]
        all_pruned_models = all_pruned_models[skip_first:]

    assert len(checkpoint_steps) == len(all_pruned_models), "Checkpoint steps and pruned models must have the same length"
    assert len(checkpoint_steps) > 0, "No checkpoint steps found"

    # Determine max run length and total units
    max_run_length = len(all_pruned_models[0]) - 1
    total_units = max_run_length

    # For each run length > 0, plot all runs (start unit) as lines, with a colormap gradient, and always include the base line
    for run_length in range(1, max_run_length + 1):
        # Collect all start keys for this run length
        start_keys = list(all_pruned_models[0][run_length].keys())
        # Robustly split start_key into unit_name and block_idx, even if unit_name contains underscores
        deconstructed_start_keys = [(int(block_idx), unit_name) for unit_name, block_idx in [start_key.rsplit('_', 1) for start_key in start_keys]]
        deconstructed_ordered_start_keys = sorted(deconstructed_start_keys)
        # Only keep runs that are not at the edges
        ordered_start_keys = [f"{unit_name}_{block_idx}" for block_idx, unit_name in deconstructed_ordered_start_keys[cut_edges:-cut_edges]]
        num_lines = len(ordered_start_keys)
        if num_lines == 0:
            continue
        colors = COLORMAP(np.linspace(0, 1, num_lines))

        plt.figure(figsize=(20, 12))
        # Plot each run (start unit)
        for idx, start_key in enumerate(ordered_start_keys):
            y = [pruned_models[run_length][start_key]["perplexity"] for pruned_models in all_pruned_models]
            plt.plot(checkpoint_steps, y, label=f"start={start_key}", color=colors[idx])
        # Plot base line
        base_y = [pruned_models[0]["base"]["perplexity"] for pruned_models in all_pruned_models]
        plt.plot(checkpoint_steps, base_y, label="base", linewidth=3, color="black", linestyle="--")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Perplexity (pruned)")
        plt.title(f"Pruned perplexity, run length {run_length}: {model_name}, {dataset_name}, maxlen={maxlen}, dtype={dtype}, cut_edges={cut_edges}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"pruned_pplx_runlen_{run_length}.png"))
        plt.close()

if __name__ == "__main__":
    arguably.run() 