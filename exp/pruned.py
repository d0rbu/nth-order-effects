import os
import yaml
import time
from dataclasses import dataclass, asdict
import itertools
from typing import TypeAlias

import arguably
import torch as th
import torch.nn as nn
from tqdm import tqdm

from core.surgical_gpt_neox import SurgicalGPTNeoXForCausalLM
from core.surgical_olmo import SurgicalOlmo2ForCausalLM
from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.gradients import compute_gradients
from core.prune import ORDERED_MODEL_UNIT_CONFIGS, prune_model, ModelUnit
from exp.exp_data import DTYPE_MAP, DATA_FILE_YAML, METADATA_FILE, PRUNED_OUT_SUBDIR

@dataclass(frozen=True)
class Datapoint:
    perplexity: float

PrunedSweepResults: TypeAlias = list[dict[str, Datapoint]]

def main(
    *args,
    model_name: str = "olmo2-7b",
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    maxlen: int = 512,
    device: str = "cuda",
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    out_dir: str = "out",
    batchsize: int = 0,
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer, checkpoint = get_model_and_tokenizer(model_name, checkpoint_idx, model_kwargs=model_kwargs)

    ordered_model_units = ORDERED_MODEL_UNIT_CONFIGS.get(type(model))
    assert ordered_model_units is not None, f"Unsupported model type: {type(model)}"

    print("Getting perplexity on training set")
    # prepare inputs for perplexity evaluation
    encoded = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].bool().to(device)
    labels = th.full_like(input_ids, -100)
    labels[attention_mask] = input_ids[attention_mask]
    dataset_size = input_ids.size(0)
    eval_batchsize = batchsize if batchsize > 0 else dataset_size

    # Build a flat list of all ModelUnits in order
    all_units = []
    for block_idx, block in enumerate(model.model.layers):
        for unit_name in ordered_model_units:
            all_units.append(ModelUnit(block_idx=block_idx, unit_name=unit_name))

    N = len(all_units)
    pruned_models: PrunedSweepResults = [{} for _ in range(N+1)]  # pruned_models[run_length][start_idx] = Datapoint

    with th.no_grad():
        # Try all contiguous runs of units (length 1, 2, ..., N)
        for run_length in tqdm(range(1, N+1), desc="Run lengths", leave=True):
            for start in tqdm(range(N - run_length + 1), desc=f"Starts (run_length={run_length})", leave=False):
                units_to_prune = all_units[start:start+run_length]
                unprune = prune_model(model, set(units_to_prune))

                # compute perplexity on pruned model and record
                loss = th.tensor(0.0, device=device)
                for batch_input_ids, batch_attention_mask, batch_labels in tqdm(zip(
                    input_ids.split(eval_batchsize),
                    attention_mask.split(eval_batchsize),
                    labels.split(eval_batchsize),
                ), total=len(input_ids) // eval_batchsize, desc=f"Batches (prune {run_length}, start {start})", leave=False):
                    batch_loss = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_labels,
                        track_activations=False,
                        num_items_in_batch=batch_attention_mask.sum(),
                    )
                    loss += batch_loss * batch_attention_mask.sum()

                mean_loss = loss / attention_mask.sum()
                pruned_ppl = float(th.exp(mean_loss))
                start_unit_key = all_units[start].key()
                pruned_models[run_length][start_unit_key] = Datapoint(perplexity=pruned_ppl)
                unprune()
    
    # compute perplexity on full model and assemble final_data
    with th.no_grad():
        loss = th.tensor(0.0, device=device)
        for batch_input_ids, batch_attention_mask, batch_labels in tqdm(zip(
            input_ids.split(eval_batchsize),
            attention_mask.split(eval_batchsize),
            labels.split(eval_batchsize),
        ), total=len(input_ids) // eval_batchsize, desc="Batches", leave=False):
            loss = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                track_activations=False,
                num_items_in_batch=batch_attention_mask.sum(),
            )
            loss += loss * batch_attention_mask.sum()

        mean_loss = loss / attention_mask.sum()
        base_ppl = float(th.exp(mean_loss))

    pruned_models[0] = {"base": Datapoint(perplexity=base_ppl)}

    # Save as list of dicts of asdict(Datapoint)
    out_timestamp_dir = str(int(time.time() * 1000))
    final_out_dir = os.path.join(out_dir, PRUNED_OUT_SUBDIR, out_timestamp_dir)

    out_filepath = os.path.join(final_out_dir, DATA_FILE_YAML)
    metadata_out_filepath = os.path.join(final_out_dir, METADATA_FILE)

    os.makedirs(final_out_dir, exist_ok=True)
    with open(out_filepath, "w") as f:
        yaml.dump([{k: asdict(v) for k, v in d.items()} for d in pruned_models], f)

    print(f"Writing metadata to {metadata_out_filepath}")
    with open(metadata_out_filepath, "w") as f:
        metadata = {
            "model": model_name,
            "dataset": dataset_name,
            "checkpoint_idx": checkpoint_idx,
            "checkpoint_metadata": {
                "step": checkpoint.step,
                "num_tokens": checkpoint.num_tokens,
                "model_config": {
                    "hf_name": checkpoint.model_config.hf_name,
                    "surgical_class": checkpoint.model_config.surgical_class.__name__,
                }
            } if checkpoint is not None else None,
            "maxlen": maxlen,
            "device": device,
            "dtype": dtype,
            "load_in_8bit": load_in_8bit,
            "load_in_4bit": load_in_4bit,
            "out_dir": out_dir,
        }

        yaml.dump(metadata, f)


if __name__ == "__main__":
    command = arguably.command(main)

    arguably.run()
