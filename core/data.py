from dataclasses import dataclass


@dataclass
class Dataset:
    content_key: str = "text"


DATASETS = {
    "ivanzhouyq/RedPajama-Tiny": Dataset(),
}
