from dataclasses import dataclass

import torch as th
import torch.nn as nn
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2RMSNorm,
    Olmo2Attention,
    Olmo2MLP,
    Olmo2DecoderLayer,
    Olmo2RotaryEmbedding,
    Olmo2PreTrainedModel,
    Olmo2Model,
    Olmo2ForCausalLM,
)


@dataclass
class SurgicalOlmo2AttentionActivations:
    query_activation: th.FloatTensor
    key_activation: th.FloatTensor
    value_activation: th.FloatTensor
    attention_map: th.FloatTensor
    attention_activation: th.FloatTensor
    output: th.FloatTensor  # o_proj(attention_activation)

@dataclass
class SurgicalOlmo2MLPActivations:
    gate_proj_activation: th.FloatTensor
    gate_proj_nonlinear_activation: th.FloatTensor
    up_proj_activation: th.FloatTensor
    hidden_activation: th.FloatTensor
    output: th.FloatTensor  # down_proj(hidden_activation)

@dataclass
class SurgicalOlmo2DecoderLayerActivations:
    attention_activations: SurgicalOlmo2AttentionActivations
    attention_normed_output: th.FloatTensor  # norm(attention_activations.output)
    mlp_activations: SurgicalOlmo2MLPActivations
    mlp_normed_output: th.FloatTensor  # norm(mlp_activations.output)

@dataclass
class SurgicalOlmo2ModelActivations:
    residual_base: th.FloatTensor
    layer_activations: list[SurgicalOlmo2DecoderLayerActivations]
    output: th.FloatTensor  # norm(final residual state)

@dataclass
class SurgicalOlmo2Output:
    loss: th.FloatTensor
    logits: th.FloatTensor
    activations: SurgicalOlmo2ModelActivations | None = None


class SurgicalOlmo2Model(Olmo2Model):
    def __init__(self, *args, **kwargs):
        # skip the Olmo2Model constructor
        super(Olmo2PreTrainedModel, self).__init__(*args, **kwargs)

        self.model 

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class SurgicalOlmo2PreTrainedModel(Olmo2PreTrainedModel):
    def __init__(self, *args, **kwargs):
        # skip the Olmo2Model constructor
        super(Olmo2PreTrainedModel, self).__init__(*args, **kwargs)

        self.model 

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class SurgicalOlmo2ForCausalLM(SurgicalOlmo2PreTrainedModel, Olmo2ForCausalLM):
    def __init__(self, config):
        SurgicalOlmo2PreTrainedModel.__init__(self, config)

        self.model = SurgicalOlmo2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        output_activations: bool = False,
    ) -> SurgicalOlmo2Output:
        pass
