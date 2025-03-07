from dataclasses import dataclass, asdict
from functools import partial
from typing import Callable
import gc
import os
import yaml
import time

import arguably
import torch as th
import torch.nn as nn
from tqdm import tqdm

from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.nth_order import compute_nth_order_deltas_backward
from exp.exp_data import DTYPE_MAP, DATA_FILE, METADATA_FILE, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR


@dataclass
class DeltaStats:
    unit_index: int
    unit_indices: list[int]
    
    # metrics to measure similarity with the original gradient
    cosine_similarity: float
    dot_product: float
    l2_distance: float
    l1_distance: float

    # metrics to measure the delta gradient itself
    l2_norm: float
    l1_norm: float

@dataclass
class UnitStats:
    unit_index: int

    # metrics to measure similarity with the original gradient
    avg_cosine_similarity_by_depth: list[float | None]
    avg_dot_product_by_depth: list[float | None]
    avg_l2_distance_by_depth: list[float | None]
    avg_l1_distance_by_depth: list[float | None]
    avg_cosine_similarity_by_unit: list[float | None]
    avg_dot_product_by_unit: list[float | None]
    avg_l2_distance_by_unit: list[float | None]
    avg_l1_distance_by_unit: list[float | None]

    # metrics to measure the delta gradient itself
    avg_l2_norm_by_depth: list[float | None]
    avg_l1_norm_by_depth: list[float | None]
    avg_l2_norm_by_unit: list[float | None]
    avg_l1_norm_by_unit: list[float | None]

def main(
    *args,
    model_name: str = "olmo2",
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    maxlen: int = 512,
    device: str = "cuda",
    dtype: str = "fp32",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    n: int = 3,
    out_dir: str = "out",
    batchsize: int = 0,
    num_samples: int = 0,
    seed: int = 44,
    sample_by_circuit: bool = False,
) -> None:
    dataset = get_dataset(dataset_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": DTYPE_MAP[dtype],
        "load_in_8bit": load_in_8bit,
        "load_in_4bit": load_in_4bit,
    }
    model, tokenizer, checkpoint = get_model_and_tokenizer(model_name, checkpoint_idx, model_kwargs=model_kwargs)

    deltas, units_deltas, units_deltas_cumulative, inputs, gradients = compute_nth_order_deltas_backward(
        model,
        checkpoint,
        tokenizer,
        dataset,
        stop_n=n,
        max_token_length=maxlen,
        batchsize=batchsize,
        num_samples=num_samples,
        seed=seed,
        sample_by_circuit=sample_by_circuit,
    )

    del deltas

    attention_mask = inputs["attention_mask"]
    num_units = len(units_deltas_cumulative)

    unit_stats = [None for _ in range(num_units)]
    all_stats = []
    for unit_idx, (unit_deltas, raw_unit_gradient) in tqdm(enumerate(zip(reversed(units_deltas_cumulative), gradients)), desc="Computing stats for units", total=num_units, leave=False):
        unit_gradient = raw_unit_gradient[attention_mask]  # T', D
        avg_cosine_similarity_by_depth = [set() for _ in range(n + 1)]
        avg_dot_product_by_depth = [set() for _ in range(n + 1)]
        avg_l2_distance_by_depth = [set() for _ in range(n + 1)]
        avg_l1_distance_by_depth = [set() for _ in range(n + 1)]
        avg_cosine_similarity_by_unit = [set() for _ in range(num_units)]
        avg_dot_product_by_unit = [set() for _ in range(num_units)]
        avg_l2_distance_by_unit = [set() for _ in range(num_units)]
        avg_l1_distance_by_unit = [set() for _ in range(num_units)]
        avg_l2_norm_by_depth = [set() for _ in range(n + 1)]
        avg_l1_norm_by_depth = [set() for _ in range(n + 1)]
        avg_l2_norm_by_unit = [set() for _ in range(num_units)]
        avg_l1_norm_by_unit = [set() for _ in range(num_units)]

        for delta_idx, nth_order_delta in tqdm(enumerate(unit_deltas), desc=f"Computing stats for unit {unit_idx}", total=len(unit_deltas), leave=False):
            raw_unit_indices = nth_order_delta.unit_indices()
            unit_indices = [num_units - 2 - unit_idx for unit_idx in raw_unit_indices]
            depth = len(unit_indices) - 1

            raw_gradient = nth_order_delta.delta.to(model.device)  # B, T, D
            gradient = raw_gradient[attention_mask]  # T', D
            avg_cosine_similarity = th.nn.functional.cosine_similarity(gradient, unit_gradient, dim=-1).mean().item()
            avg_dot_product = (gradient * unit_gradient).sum(dim=-1).mean().item()
            avg_l2_distance = th.nn.functional.pairwise_distance(gradient, unit_gradient, p=2).mean().item()
            avg_l1_distance = th.nn.functional.pairwise_distance(gradient, unit_gradient, p=1).mean().item()
            avg_l2_norm = th.norm(gradient, p=2, dim=-1).mean().item()
            avg_l1_norm = th.norm(gradient, p=1, dim=-1).mean().item()

            stats = DeltaStats(
                unit_index=unit_idx,
                unit_indices=unit_indices,
                cosine_similarity=avg_cosine_similarity,
                dot_product=avg_dot_product,
                l2_distance=avg_l2_distance,
                l1_distance=avg_l1_distance,
                l2_norm=avg_l2_norm,
                l1_norm=avg_l1_norm,
            )
            all_stats.append(stats)

            for contained_unit_idx in unit_indices:
                avg_cosine_similarity_by_unit[contained_unit_idx].add(avg_cosine_similarity)
                avg_dot_product_by_unit[contained_unit_idx].add(avg_dot_product)
                avg_l2_distance_by_unit[contained_unit_idx].add(avg_l2_distance)
                avg_l1_distance_by_unit[contained_unit_idx].add(avg_l1_distance)
                avg_l2_norm_by_unit[contained_unit_idx].add(avg_l2_norm)
                avg_l1_norm_by_unit[contained_unit_idx].add(avg_l1_norm)

            avg_cosine_similarity_by_depth[depth].add(avg_cosine_similarity)
            avg_dot_product_by_depth[depth].add(avg_dot_product)
            avg_l2_distance_by_depth[depth].add(avg_l2_distance)
            avg_l1_distance_by_depth[depth].add(avg_l1_distance)
            avg_l2_norm_by_depth[depth].add(avg_l2_norm)
            avg_l1_norm_by_depth[depth].add(avg_l1_norm)

            del raw_gradient, gradient
            th.cuda.empty_cache()

        avg_cosine_similarity_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_cosine_similarity_by_depth]
        avg_dot_product_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_dot_product_by_depth]
        avg_l2_distance_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l2_distance_by_depth]
        avg_l1_distance_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l1_distance_by_depth]
        avg_cosine_similarity_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_cosine_similarity_by_unit]
        avg_dot_product_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_dot_product_by_unit]
        avg_l2_distance_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l2_distance_by_unit]
        avg_l1_distance_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l1_distance_by_unit]
        avg_l2_norm_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l2_norm_by_depth]
        avg_l1_norm_by_depth = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l1_norm_by_depth]
        avg_l2_norm_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l2_norm_by_unit]
        avg_l1_norm_by_unit = [sum(x) / len(x) if len(x) > 0 else None for x in avg_l1_norm_by_unit]

        unit_stats[unit_idx] = UnitStats(
            unit_index=unit_idx,
            avg_cosine_similarity_by_depth=avg_cosine_similarity_by_depth,
            avg_dot_product_by_depth=avg_dot_product_by_depth,
            avg_l2_distance_by_depth=avg_l2_distance_by_depth,
            avg_l1_distance_by_depth=avg_l1_distance_by_depth,
            avg_cosine_similarity_by_unit=avg_cosine_similarity_by_unit,
            avg_dot_product_by_unit=avg_dot_product_by_unit,
            avg_l2_distance_by_unit=avg_l2_distance_by_unit,
            avg_l1_distance_by_unit=avg_l1_distance_by_unit,
            avg_l2_norm_by_depth=avg_l2_norm_by_depth,
            avg_l1_norm_by_depth=avg_l1_norm_by_depth,
            avg_l2_norm_by_unit=avg_l2_norm_by_unit,
            avg_l1_norm_by_unit=avg_l1_norm_by_unit,
        )

        gradients[unit_idx] = None
        del unit_gradient
        th.cuda.empty_cache()

    del model, tokenizer, dataset, inputs, gradients
    th.cuda.empty_cache()
    gc.collect()

    final_data_all_stats = [asdict(stat) for stat in all_stats]
    final_data_unit_stats = [asdict(stat) for stat in unit_stats]
    final_data = {
        "all_stats": final_data_all_stats,
        "unit_stats": final_data_unit_stats,
    }

    out_timestamp_dir = str(int(time.time() * 1000))
    final_out_dir = os.path.join(out_dir, BACKWARD_CONTRIBUTIONS_OUT_SUBDIR, out_timestamp_dir)

    out_filepath = os.path.join(final_out_dir, DATA_FILE)
    metadata_out_filepath = os.path.join(final_out_dir, METADATA_FILE)

    os.makedirs(final_out_dir, exist_ok=True)
    with open(out_filepath, "w") as f:
        yaml.dump(final_data, f)

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
            "n": n,
            "out_dir": out_dir,
            "batchsize": batchsize,
            "num_samples": num_samples,
            "seed": seed,
            "sample_by_circuit": sample_by_circuit,
        }

        yaml.dump(metadata, f)


if __name__ == "__main__":
    command = arguably.command(main)

    arguably.run()
