import re
from dataclasses import dataclass
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from huggingface_hub import list_repo_refs
import torch as th

from core.surgical_olmo import SurgicalOlmo2ForCausalLM


MODEL_NAME = "allenai/OLMo-2-1124-7B"
BRANCH_REGEX = re.compile(r"stage1-step(\d+)-tokens(\d+)B")
REVISION_FORMAT = "step{}-tokens{}B"


@dataclass
class Checkpoint:
    step: int
    num_tokens: int

    def __str__(self):
        return REVISION_FORMAT.format(self.step, self.num_tokens)

    def load_model(self, model_kwargs: dict[str, Any]) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=str(self), **model_kwargs)


refs = list_repo_refs(MODEL_NAME)
all_branches = [branch.name for branch in refs.branches]

stage1_checkpoints = []
for branch in all_branches:
    match = re.match(BRANCH_REGEX, branch)
    if not match:
        continue

    step, num_tokens = match.groups()
    stage1_checkpoints.append(Checkpoint(int(step), int(num_tokens)))


def get_model_and_tokenizer(checkpoint_idx: int | None = None, model_kwargs: dict[str, Any] = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if checkpoint_idx is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    else:
        model = stage1_checkpoints[checkpoint_idx].load_model(model_kwargs)

    # clone the model to the surgical version by getting and setting the state dict
    surgical_model = SurgicalOlmo2ForCausalLM.from_olmo2_for_causal_lm(model)

    return surgical_model, tokenizer
