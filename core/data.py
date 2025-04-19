from dataclasses import dataclass

from datasets import load_dataset
from typing import Iterable
from typing import Callable
from itertools import chain


@dataclass
class DatasetConfig:
    name: str = "ivanzhouyq/RedPajama-Tiny"
    split: str = "train"
    content_key: str = "text"
    selection: Callable[[], Iterable[int]] | None = None


DATASETS = {
    "redpajama-tiny": DatasetConfig(),
    "redpajama-micro": DatasetConfig(
        # gets the first 16 samples from each source
        selection=lambda: chain(*[range(source_idx * 64, source_idx * 64 + 16) for source_idx in range(7)])
    ),
    "redpajama-nano": DatasetConfig(
        # gets the first 2 samples from each source
        # 0, 1, 64, 65, 128, 129...
        selection=lambda: chain(*[range(source_idx * 64, source_idx * 64 + 2) for source_idx in range(7)])
    ),
    "redpajama-pico": DatasetConfig(
        # gets the first sample from each source
        # 0, 64, 128...
        selection=lambda: range(0, 64 * 7, 64)
    ),
    "redpajama-1": DatasetConfig(
        selection=lambda: [139]  # selection comes from https://www.sensory.com/category/security/biometrics-security/
    ),
}


def get_dataset(name: str) -> list[str]:
    assert name in DATASETS, f"Dataset {name} not found in DATASETS"

    config = DATASETS[name]

    dataset = load_dataset(config.name, split=config.split)

    if config.selection is not None:
        dataset = dataset.select(config.selection())

    return dataset[config.content_key]
