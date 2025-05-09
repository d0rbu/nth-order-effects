import os
import yaml
import arguably
import matplotlib.pyplot as plt
from tqdm import tqdm

from exp.exp_data import get_exp_data, PRUNED_OUT_SUBDIR, DATA_FILE_YAML
from core.model import MODELS

FIGURES_DIR = "figures/pruned_pplx"

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

    # Gather all pruned perplexities by layer/unit for each checkpoint
    all_layer_unit_ppl = dict()  # (block_idx, unit_name) -> [ppl at each checkpoint]
    checkpoint_steps = []
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments), leave=False):
        data_path = os.path.join(exp_path, DATA_FILE_YAML)
        with open(data_path, "r") as f:
            data = yaml.safe_load(f)
        checkpoint_steps.append(experiment.checkpoint_step)
        pruned_models = data["pruned_models"]
        for unit_key, dp in pruned_models.items():
            if unit_key not in all_layer_unit_ppl:
                all_layer_unit_ppl[unit_key] = []
            all_layer_unit_ppl[unit_key].append(dp["perplexity"])

    # Group unit_keys by layer
    layer_to_unitkeys = dict()
    for unit_key in all_layer_unit_ppl:
        block_idx, unit_name = unit_key.split("_", 1)
        if block_idx not in layer_to_unitkeys:
            layer_to_unitkeys[block_idx] = []
        layer_to_unitkeys[block_idx].append(unit_key)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for block_idx, unit_keys in layer_to_unitkeys.items():
        plt.figure(figsize=(10, 6))
        for unit_key in unit_keys:
            unit_name = unit_key.split("_", 1)[1]
            plt.plot(checkpoint_steps, all_layer_unit_ppl[unit_key], label=f"{unit_name}")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Perplexity (pruned)")
        plt.title(f"Layer {block_idx} pruned perplexity by unit")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"layer_{block_idx}_pruned_pplx.png"))
        plt.close()

if __name__ == "__main__":
    arguably.run() 