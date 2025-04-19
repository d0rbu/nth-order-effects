import os
import json

import arguably
import torch as th
from tqdm import tqdm

from core.data import get_dataset
from core.model import get_tokenizer
from exp.contributions import DATA_FILE
from exp.exp_data import get_exp_data, GRADIENT_OUT_SUBDIR


JSON_DIR = "json_gradient"


def json_key(
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
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-1",
    maxlen: int = 256,
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    out_dir: str = "out",
) -> None:
    completed_experiments = get_exp_data(out_dir, GRADIENT_OUT_SUBDIR)
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

    all_gradients = []  # T', B, T, U
    attention_mask = None  # B, T
    for experiment, exp_path in tqdm(sorted_experiments, desc="Loading data", total=len(sorted_experiments), leave=False):
        data_path = os.path.join(exp_path, DATA_FILE)
        data = th.load(data_path)
        # get norm
        gradients = th.linalg.vector_norm(data["gradients"], dim=-1)  # U, B, T
        attention_mask = data["attention_mask"]  # B, T

        gradients = gradients.permute(1, 2, 0)  # B, T, U

        all_gradients.append(gradients)

    all_distributions = th.stack(all_gradients, dim=3)  # B, T, U, T'

    # for this case, visualization means exporting to json so we can see it in the frontend
    out_dir = os.path.join(out_dir, JSON_DIR)
    os.makedirs(out_dir, exist_ok=True)

    out_filepath = os.path.join(out_dir, f"{json_key(model_name, dataset_name, maxlen, dtype, load_in_8bit, load_in_4bit)}.json")

    gradients_list = [None for _ in range(all_distributions.shape[0])]
    for batch_idx, (sample_gradients, sample_mask) in tqdm(enumerate(zip(all_distributions, attention_mask)), desc="Processing gradients", total=all_distributions.shape[0], leave=False):
        sequence_length = sample_mask.sum().item()
        gradients_list[batch_idx] = sample_gradients[:sequence_length].tolist()

    tokenizer = get_tokenizer(model_name)
    dataset = get_dataset(dataset_name)

    # make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = [tokenizer.tokenize(text, truncation=True, max_length=maxlen) for text in tqdm(dataset, desc="Tokenizing dataset", leave=False)]

    out = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "maxlen": maxlen,
        "dtype": dtype,
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
        "gradients": gradients_list,
        "dataset": dataset,
        "tokenized_dataset": tokenized_dataset,
    }

    with open(out_filepath, "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    arguably.run()
