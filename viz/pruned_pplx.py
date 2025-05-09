import os
import yaml
import arguably
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp.exp_data import get_exp_data, PRUNED_OUT_SUBDIR, DATA_FILE_YAML
from core.model import MODELS

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

    model_config = MODELS[model_name]
    ordered_units = model_config.surgical_class.unit_names if hasattr(model_config.surgical_class, 'unit_names') else None

    # Gather all pruned perplexities by unit for each checkpoint
    all_unit_ppl = dict()  # unit_key -> [ppl at each checkpoint]
    base_ppl = []
    checkpoint_steps = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments), leave=False):
        data_path = os.path.join(exp_path, DATA_FILE_YAML)
        with open(data_path, "r") as f:
            data = yaml.safe_load(f)
        checkpoint_steps.append(experiment.checkpoint_step)
        pruned_models = data["pruned_models"]
        for unit_key, dp in pruned_models.items():
            if unit_key not in all_unit_ppl:
                all_unit_ppl[unit_key] = []
            all_unit_ppl[unit_key].append(dp["perplexity"])
        # base model
        base_ppl.append(data["base_model"]["perplexity"])

    # Optionally skip the first n datapoints
    if skip_first > 0:
        checkpoint_steps = checkpoint_steps[skip_first:]
        for unit_key in all_unit_ppl:
            all_unit_ppl[unit_key] = all_unit_ppl[unit_key][skip_first:]
        base_ppl = base_ppl[skip_first:]

    # Sort unit_keys for consistent legend (by block then unit)
    def unit_sort_key(unit_key):
        block_idx, unit_name = unit_key.split("_", 1)
        # try to sort attn before mlp if possible
        return (int(block_idx), unit_name)
    sorted_unit_keys = sorted(all_unit_ppl.keys(), key=unit_sort_key)

    key = figure_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit)
    figures_dir = os.path.join("figures_pruned", key)
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(20, 12))
    all_values = []
    for unit_key in sorted_unit_keys:
        block_idx, unit_name = unit_key.split("_", 1)
        label = f"{unit_name}_{block_idx}"
        values = np.array(all_unit_ppl[unit_key])
        all_values.append(values)
        plt.plot(checkpoint_steps, values, label=label)
    # Add base model line
    base_ppl_arr = np.array(base_ppl)
    all_values.append(base_ppl_arr)
    plt.plot(checkpoint_steps, base_ppl_arr, label="base", linewidth=3, color="black", linestyle="--")
    plt.xlabel("Checkpoint step")
    plt.ylabel("Perplexity (pruned)")
    plt.title(f"Pruned perplexity by unit and base: {model_name}, {dataset_name}, maxlen={maxlen}, dtype={dtype}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, f"pruned_pplx_all_units.png"))
    plt.close()

if __name__ == "__main__":
    arguably.run() 