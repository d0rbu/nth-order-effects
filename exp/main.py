import arguably

from core.data import get_dataset
from core.model import get_model_and_tokenizer
from core.nth_order import nth_order_deltas


@arguably.command
def main(
    *args,
    dataset_name: str = "redpajama-nano",
    checkpoint_idx: int | None = None,
    n: int = 1,
):
    dataset = get_dataset(dataset_name)
    model, tokenizer = get_model_and_tokenizer(checkpoint_idx)

    deltas = nth_order_deltas(model, tokenizer, dataset, n=n)


if __name__ == "__main__":
    arguably.run()
