import re
from dataclasses import dataclass
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.logging import disable_progress_bar
from huggingface_hub import list_repo_refs
import torch as th

from core.surgical_olmo import SurgicalOlmo2ForCausalLM
from core.surgical_gpt_neox import SurgicalGPTNeoXForCausalLM


SurgicalModel = SurgicalOlmo2ForCausalLM | SurgicalGPTNeoXForCausalLM


@dataclass
class Checkpoint:
    step: int
    num_tokens: int
    model_config: "ModelConfig"

    def __str__(self):
        return self.model_config.revision_format.format(self.step, self.num_tokens)

    def load_model(self, model_kwargs: dict[str, Any]) -> PreTrainedModel:
        print("Loading model", self)
        model = AutoModelForCausalLM.from_pretrained(self.model_config.hf_name, revision=str(self), **model_kwargs)
        print("Model loaded")

        return model

@dataclass
class ModelConfig:
    hf_name: str
    branch_regex: re.Pattern
    revision_format: str
    surgical_class: th.nn.Module
    tokenizer_has_padding_token: bool = True
    checkpoints: list[Checkpoint] = None

    def __post_init__(self):
        self.branch_regex = re.compile(self.branch_regex)

        refs = list_repo_refs(self.hf_name)
        self.all_branches = [branch.name for branch in refs.branches]

        checkpoints = []
        for branch in self.all_branches:
            match = re.match(self.branch_regex, branch)
            if not match:
                continue

            groups = match.groups()
            if len(groups) == 2:
                step, num_tokens = groups
            elif len(groups) == 1:
                step = groups[0]
                num_tokens = 0
            else:
                raise ValueError(f"Unexpected number of groups in branch {branch}")

            checkpoints.append(Checkpoint(int(step), int(num_tokens), self))

        self.checkpoints = sorted(checkpoints, key=lambda x: x.step)

MODELS = {
    "olmo2-1b": ModelConfig(
        hf_name="allenai/OLMo-2-0425-1B",
        branch_regex=re.compile(r"stage1-step(\d+)-tokens(\d+)B"),
        revision_format="stage1-step{}-tokens{}B",
        surgical_class=SurgicalOlmo2ForCausalLM,
    ),
    "olmo2-7b": ModelConfig(
        hf_name="allenai/OLMo-2-1124-7B",
        branch_regex=re.compile(r"stage1-step(\d+)-tokens(\d+)B"),
        revision_format="stage1-step{}-tokens{}B",
        surgical_class=SurgicalOlmo2ForCausalLM,
    ),
    "olmo2-13b": ModelConfig(
        hf_name="allenai/OLMo-2-1124-13B",
        branch_regex=re.compile(r"stage1-step(\d+)-tokens(\d+)B"),
        revision_format="stage1-step{}-tokens{}B",
        surgical_class=SurgicalOlmo2ForCausalLM,
    ),
    "olmo2-32b": ModelConfig(
        hf_name="allenai/OLMo-2-0325-32B",
        branch_regex=re.compile(r"stage1-step(\d+)-tokens(\d+)B"),
        revision_format="stage1-step{}-tokens{}B",
        surgical_class=SurgicalOlmo2ForCausalLM,
    ),
    "pythia14m": ModelConfig(
        hf_name="EleutherAI/pythia-14m",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia31m": ModelConfig(
        hf_name="EleutherAI/pythia-31m",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia70m": ModelConfig(
        hf_name="EleutherAI/pythia-70m-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia160m": ModelConfig(
        hf_name="EleutherAI/pythia-160m-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia410m": ModelConfig(
        hf_name="EleutherAI/pythia-410m-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia1b": ModelConfig(
        hf_name="EleutherAI/pythia-1b-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia1.4b": ModelConfig(
        hf_name="EleutherAI/pythia-1.4b-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia2.8b": ModelConfig(
        hf_name="EleutherAI/pythia-2.8b-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia6.9b": ModelConfig(
        hf_name="EleutherAI/pythia-6.9b-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
    "pythia12b": ModelConfig(
        hf_name="EleutherAI/pythia-12b-deduped",
        branch_regex=re.compile(r"step(\d+)"),
        revision_format="step{}",
        surgical_class=SurgicalGPTNeoXForCausalLM,
        tokenizer_has_padding_token=False,
    ),
}

disable_progress_bar()

def get_tokenizer(model_name: str = "olmo2-7b") -> PreTrainedTokenizerBase:
    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found in MODELS. Available models: {list(MODELS.keys())}"

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_name, use_fast=True)

    if not model_config.tokenizer_has_padding_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_and_tokenizer(model_name: str = "olmo2-7b", checkpoint_idx: int | None = None, model_kwargs: dict[str, Any] = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, Checkpoint | None]:
    assert (model_config := MODELS.get(model_name, None)), f"Model {model_name} not found in MODELS. Available models: {list(MODELS.keys())}"

    tokenizer = get_tokenizer(model_name)

    checkpoint = None
    step = None

    if checkpoint_idx is None:
        model = AutoModelForCausalLM.from_pretrained(model_config.hf_name, **model_kwargs)
    else:
        checkpoint = model_config.checkpoints[checkpoint_idx]
        step = checkpoint.step
        model = checkpoint.load_model(model_kwargs)

    # clone the model to the surgical version by getting and setting the state dict
    surgical_model = model_config.surgical_class.from_causal_lm(model)

    return surgical_model, tokenizer, checkpoint
