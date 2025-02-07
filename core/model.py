import re
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from huggingface_hub import list_repo_refs

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

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=str(self))


refs = list_repo_refs(MODEL_NAME)
all_branches = [branch.name for branch in refs.branches]

stage1_checkpoints = []
for branch in all_branches:
    match = re.match(BRANCH_REGEX, branch)
    if not match:
        continue

    step, num_tokens = match.groups()
    stage1_checkpoints.append(Checkpoint(int(step), int(num_tokens)))


def get_model_and_tokenizer(checkpoint_idx: int | None = None) -> tuple[SurgicalOlmo2ForCausalLM, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if checkpoint_idx is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    else:
        model = stage1_checkpoints[checkpoint_idx].load_model()

    # clone the model to the surgical version by getting and setting the state dict
    surgical_model = SurgicalOlmo2ForCausalLM(model.config)
    surgical_model.load_state_dict(model.state_dict())

    return surgical_model, tokenizer
