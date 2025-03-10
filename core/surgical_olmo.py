import itertools
from typing import Callable

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
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger
)
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from core.activations import (
    CausalLMActivations,
    ModelActivations,
    DecoderLayerActivations,
    AttentionActivations,
    MLPActivations,
)


class SurgicalOlmo2RMSNorm(Olmo2RMSNorm):
    pass

class SurgicalOlmo2Attention(Olmo2Attention):
    def __init__(self, config: Olmo2Config, layer_idx: int | None = None):
        super(Olmo2Attention, self).__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = SurgicalOlmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = SurgicalOlmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: th.FloatTensor,
        position_embeddings: th.FloatTensor | None = None,
        attention_mask: th.BoolTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
        track_activations: bool = True,
        **kwargs,
    ) -> AttentionActivations | th.FloatTensor:
        if track_activations and activation_mask:
            return self.forward_track_activation(
                hidden_states,
                position_embeddings,
                attention_mask,
                activation_mask,
                **kwargs,
            )
        else:
            return self.forward_no_activation(
                hidden_states,
                position_embeddings,
                attention_mask,
                **kwargs,
            )

    def forward_track_activation(
        self,
        hidden_states: th.FloatTensor,
        position_embeddings: th.FloatTensor | None = None,
        attention_mask: th.BoolTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
        **kwargs,
    ) -> AttentionActivations:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        normed_query_states = self.q_norm(query_states)
        normed_key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        normed_query_states = normed_query_states.view(hidden_shape).transpose(1, 2)
        normed_key_states = normed_key_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        rotated_query_states, rotated_key_states = apply_rotary_pos_emb(normed_query_states, normed_key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and "attention_map" in activation_mask:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attention_output, attention_map = attention_interface(
            self,
            rotated_query_states,
            rotated_key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        contiguous_attention_output = attention_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(contiguous_attention_output)

        activations = AttentionActivations(
            query_activation=query_states,
            key_activation=key_states,
            value_activation=value_states,
            normed_query_activation=normed_query_states,
            normed_key_activation=normed_key_states,
            rotated_query_activation=rotated_query_states,
            rotated_key_activation=rotated_key_states,
            attention_map=attention_map,
            attention_output=attention_output,
            output=output,
        )
        activations.apply_activation_mask(activation_mask)

        return activations

    def forward_no_activation(
        self,
        hidden_states: th.FloatTensor,
        position_embeddings: th.FloatTensor | None = None,
        attention_mask: th.BoolTensor | None = None,
        **kwargs,
    ) -> th.FloatTensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )[0].reshape(*input_shape, -1).contiguous()

        output = self.o_proj(output)

        return output

class SurgicalOlmo2MLP(Olmo2MLP):
    def __init__(self, config: Olmo2Config):
        super(Olmo2MLP, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: th.FloatTensor,
        activation_mask: bool | list[str] = ["output"],
        track_activations: bool = True,
        **kwargs,
    ) -> MLPActivations | th.FloatTensor:
        if track_activations:
            return self.forward_track_activation(hidden_states, activation_mask)
        else:
            return self.forward_no_activation(hidden_states)

    def forward_track_activation(
        self,
        hidden_states: th.FloatTensor,
        activation_mask: bool | list[str] = ["output"],
    ) -> MLPActivations:
        gate_proj_activation = self.gate_proj(hidden_states)
        gate_proj_nonlinear_activation = self.act_fn(gate_proj_activation)
        up_proj_activation = self.up_proj(hidden_states)
        hidden_activation = gate_proj_nonlinear_activation * up_proj_activation
        output = self.down_proj(hidden_activation)

        activations = MLPActivations(
            gate_proj_activation=gate_proj_activation,
            gate_proj_nonlinear_activation=gate_proj_nonlinear_activation,
            up_proj_activation=up_proj_activation,
            hidden_activation=hidden_activation,
            output=output,
        )
        activations.apply_activation_mask(activation_mask)

        return activations

    def forward_no_activation(
        self,
        hidden_states: th.FloatTensor,
    ) -> th.FloatTensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

class SurgicalOlmo2DecoderLayer(Olmo2DecoderLayer):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super(Olmo2DecoderLayer, self).__init__()

        self.hidden_size = config.hidden_size

        self.self_attn = SurgicalOlmo2Attention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = SurgicalOlmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SurgicalOlmo2MLP(config)
        self.post_feedforward_layernorm = SurgicalOlmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def attn_unit_forward(
        self,
        hidden_states: th.FloatTensor,
        attention_mask: th.BoolTensor | None = None,
        position_embeddings: th.FloatTensor | None = None,
        **kwargs,
    ) -> th.FloatTensor:
        return self.post_attention_layernorm(
            self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        )

    def mlp_unit_forward(
        self,
        hidden_states: th.FloatTensor,
        **kwargs,
    ) -> th.FloatTensor:
        return self.post_feedforward_layernorm(self.mlp(hidden_states, **kwargs))

    def forward(
        self,
        hidden_states: th.FloatTensor,
        attention_mask: th.BoolTensor | None = None,
        position_embeddings: th.FloatTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
        track_activations: bool = True,
        **kwargs,
    ) -> DecoderLayerActivations | th.FloatTensor:
        if track_activations and activation_mask:
            return self.forward_track_activation(
                hidden_states,
                attention_mask,
                position_embeddings,
                activation_mask,
                **kwargs,
            )
        else:
            return self.forward_no_activation(
                hidden_states,
                attention_mask,
                position_embeddings,
                **kwargs,
            )

    def forward_track_activation(
        self,
        hidden_states: th.FloatTensor,
        attention_mask: th.BoolTensor | None = None,
        position_embeddings: th.FloatTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
    ) -> DecoderLayerActivations:
        residual = hidden_states
        activation_mask_for_attention = [".".join(activation_path.split(".")[1:]) for activation_path in activation_mask if activation_path.startswith("attention_activations.")]

        attention_output = self.self_attn(
            hidden_states=residual,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activation_mask=activation_mask_for_attention,
        )
        if isinstance(attention_output, AttentionActivations):
            attention_activations = attention_output
        else:
            attention_activations = AttentionActivations(output=attention_output)

        attention_normed_output = self.post_attention_layernorm(attention_activations.output)

        residual += attention_normed_output
        attention_unit_output = residual
        activation_mask_for_mlp = [".".join(activation_path.split(".")[1:]) for activation_path in activation_mask if activation_path.startswith("mlp_activations.")]

        mlp_output = self.mlp(
            residual,
            activation_mask=activation_mask_for_mlp,
        )
        if isinstance(mlp_output, MLPActivations):
            mlp_activations = mlp_output
        else:
            mlp_activations = MLPActivations(output=mlp_output)

        mlp_normed_output = self.post_feedforward_layernorm(mlp_activations.output)

        residual += mlp_normed_output

        activations = DecoderLayerActivations(
            attention_activations=attention_activations,
            attention_normed_output=attention_normed_output,
            attention_output=attention_unit_output,
            mlp_activations=mlp_activations,
            mlp_normed_output=mlp_normed_output,
            output=residual,
        )
        activations.apply_activation_mask(activation_mask)

        return activations

    def forward_no_activation(
        self,
        hidden_states: th.FloatTensor,
        attention_mask: th.BoolTensor | None = None,
        position_embeddings: th.FloatTensor | None = None,
    ) -> th.FloatTensor:
        residual = hidden_states

        attention_output = self.self_attn(
            hidden_states=residual,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            track_activations=False,
        )
        residual += self.post_attention_layernorm(attention_output)

        residual += self.post_feedforward_layernorm(self.mlp(residual, track_activations=False))

        return residual

class SurgicalOlmo2RotaryEmbedding(Olmo2RotaryEmbedding):
    pass

class SurgicalOlmo2PreTrainedModel(Olmo2PreTrainedModel):
    pass

class SurgicalOlmo2Model(SurgicalOlmo2PreTrainedModel, Olmo2Model):
    def __init__(self, config: Olmo2Config):
        SurgicalOlmo2PreTrainedModel.__init__(self, config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SurgicalOlmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = SurgicalOlmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SurgicalOlmo2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def unit_forwards(self) -> list[Callable]:
        attn_and_mlp_forwards = [(layer.attn_unit_forward, layer.mlp_unit_forward) for layer in self.layers]
        # flatten
        return list(itertools.chain(*attn_and_mlp_forwards))

    def forward(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
        track_activations: bool = True,
        **kwargs,
    ) -> ModelActivations | th.FloatTensor:
        if track_activations and activation_mask:
            return self.forward_track_activation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                activation_mask=activation_mask,
                **kwargs,
            )
        else:
            return self.forward_no_activation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

    def forward_track_activation(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        activation_mask: bool | list[str] = ["output"],
        **kwargs,
    ) -> ModelActivations:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids = th.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device, dtype=th.long
        )
        causal_mask = self._update_causal_mask(
            None, inputs_embeds, position_ids, None, True
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids.unsqueeze(0))

        layer_activations = []

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            activation_mask_for_layer = [
                ".".join(activation_path.split(".")[2:]) for activation_path in activation_mask
                if activation_path.startswith(f"layer_activations.{layer_idx}.") or activation_path.startswith("layer_activations.*.")
            ]

            if self.gradient_checkpointing and self.training:
                layer_output = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    activation_mask=activation_mask_for_layer,
                )
            else:
                layer_output = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    activation_mask=activation_mask_for_layer,
                )

            if isinstance(layer_output, DecoderLayerActivations):
                layer_activation = layer_output
                layer_output = layer_activation.output
            else:
                layer_activation = DecoderLayerActivations()

            layer_activations.append(layer_activation)

            hidden_states = layer_output

        hidden_states = self.norm(hidden_states)

        activations = ModelActivations(
            residual_base=inputs_embeds,
            layer_activations=layer_activations,
            output=hidden_states,
        )
        activations.apply_activation_mask(activation_mask)

        return activations

    def forward_no_activation(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        **kwargs,
    ) -> th.FloatTensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_ids = th.arange(
            0, inputs_embeds.shape[1], device=inputs_embeds.device, dtype=th.long
        )
        causal_mask = self._update_causal_mask(
            None, inputs_embeds, position_ids, None, True
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids.unsqueeze(0))

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    track_activations=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    track_activations=False,
                )

        return self.norm(hidden_states)

class SurgicalOlmo2ForCausalLM(SurgicalOlmo2PreTrainedModel, Olmo2ForCausalLM):
    def __init__(self, config: Olmo2Config):
        SurgicalOlmo2PreTrainedModel.__init__(self, config)

        self.model = SurgicalOlmo2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        self.post_init()

    @classmethod
    def from_causal_lm(cls, model: Olmo2ForCausalLM) -> "SurgicalOlmo2ForCausalLM":
        with th.device("meta"):
            surgical_model = cls(model.config)

        surgical_model.lm_head = model.lm_head
        surgical_model.model.rotary_emb = model.model.rotary_emb
        surgical_model.model.norm = model.model.norm
        for surgical_layer, model_layer in zip(surgical_model.model.layers, model.model.layers):
            surgical_layer.post_feedforward_layernorm = model_layer.post_feedforward_layernorm

            surgical_layer.mlp.down_proj = model_layer.mlp.down_proj
            surgical_layer.mlp.up_proj = model_layer.mlp.up_proj
            surgical_layer.mlp.gate_proj = model_layer.mlp.gate_proj

            surgical_layer.post_attention_layernorm = model_layer.post_attention_layernorm

            surgical_layer.self_attn.k_norm = model_layer.self_attn.k_norm
            surgical_layer.self_attn.q_norm = model_layer.self_attn.q_norm
            surgical_layer.self_attn.o_proj = model_layer.self_attn.o_proj
            surgical_layer.self_attn.v_proj = model_layer.self_attn.v_proj
            surgical_layer.self_attn.k_proj = model_layer.self_attn.k_proj
            surgical_layer.self_attn.q_proj = model_layer.self_attn.q_proj
        surgical_model.model.embed_tokens = model.model.embed_tokens

        return surgical_model

    def forward(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        activation_mask: bool | list[str] = ["logits", "loss"],
        track_activations: bool = True,
        **kwargs,
    ) -> CausalLMActivations | th.FloatTensor:
        if track_activations and activation_mask:
            return self.forward_track_activation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                activation_mask=activation_mask,
                **kwargs,
            )
        else:
            return self.forward_no_activation(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )

    def forward_track_activation(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        activation_mask: bool | list[str] = ["logits", "loss"],
        **kwargs,
    ) -> CausalLMActivations:
        activation_mask_for_model = [".".join(activation_path.split(".")[1:]) for activation_path in activation_mask if activation_path.startswith("model_activations.")]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        model_output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            activation_mask=activation_mask_for_model,
            **kwargs,
        )

        if isinstance(model_output, ModelActivations):
            model_activations = model_output
        else:
            model_activations = ModelActivations(output=model_output)

        logits = self.lm_head(model_activations.output)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs) if labels is not None else None

        activations = CausalLMActivations(
            logits=logits,
            model_activations=model_activations,
            loss=loss,
        )
        activations.apply_activation_mask(activation_mask)

        return activations

    def forward_no_activation(
        self,
        input_ids: th.LongTensor | None = None,
        inputs_embeds: th.FloatTensor | None = None,
        labels: th.LongTensor | None = None,
        **kwargs,
    ) -> th.FloatTensor:
        model_output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            track_activations=False,
            **kwargs,
        )

        logits = self.lm_head(model_output)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs) if labels is not None else None

        return loss
