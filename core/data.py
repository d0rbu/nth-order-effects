from dataclasses import dataclass

from datasets import load_dataset
from typing import Iterable
from itertools import chain


@dataclass
class DatasetConfig:
    name: str = "ivanzhouyq/RedPajama-Tiny"
    split: str = "train"
    content_key: str = "text"
    selection: Iterable[int] | None = None


DATASETS = {
    "redpajama-tiny": DatasetConfig(),
    "redpajama-nano": DatasetConfig(
        # gets the first 2 samples from each source
        # 0, 1, 64, 65, 128, 129...
        selection=chain(*[range(source_idx * 64, source_idx * 64 + 2) for source_idx in range(7)])
    ),
    "redpajama-1": DatasetConfig(
        selection=range(1)
    ),
}


def get_dataset(name: str) -> list[str]:
    assert name in DATASETS, f"Dataset {name} not found in DATASETS"

    config = DATASETS[name]

    dataset = load_dataset(config.name, split=config.split)

    if config.selection is not None:
        dataset = dataset.select(config.selection)

    return dataset[config.content_key]
