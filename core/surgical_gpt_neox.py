import itertools
from typing import Callable

import torch as th
import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXMLP,
    GPTNeoXLayer,
    GPTNeoXRotaryEmbedding,
    GPTNeoXPreTrainedModel,
    GPTNeoXModel,
    GPTNeoXForCausalLM,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger
)
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from core.activations import (
    CausalLMActivations,
    ModelActivations,
    DecoderLayerActivations,
    AttentionActivations,
    MLPActivations,
)

class SurgicalGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config: GPTNeoXConfig, layer_idx: int | None = None):
        super(GPTNeoXAttention, self).__init__()

        self.config = config
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.scaling = self.head_size**-0.5
        self.is_causal = True
        self.layer_idx = layer_idx

        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

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
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        rotated_query_states, rotated_key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation in ("sdpa", "flash_attention_2") and "attention_map" in activation_mask:
                logger.warning_once(
                    f"Setting `attention_type` to `eager` because `{self.config._attn_implementation}` does not support"
                    f" `output_attentions=True` or `head_mask`."
                )
            elif self.training and self.attention_dropout > 0 and self.config._attn_implementation == "flex_attention":
                logger.warning_once(
                    f"Setting `attention_type` to `eager` because `dropout` is not supported in `{self.config._attn_implementation}`."
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
        output = self.dense(contiguous_attention_output)

        activations = AttentionActivations(
            query_activation=query_states,
            key_activation=key_states,
            value_activation=value_states,
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
        hidden_shape = (*input_shape, -1, self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.training and self.attention_dropout > 0 and self.config._attn_implementation == "flex_attention":
                logger.warning_once(
                    f"Setting `attention_type` to `eager` because `dropout` is not supported in `{self.config._attn_implementation}`."
                )
            else:
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

        output = self.dense(output)

        return output

class SurgicalGPTNeoXMLP(GPTNeoXMLP):
    def __init__(self, config: GPTNeoXConfig):
        super(GPTNeoXMLP, self).__init__()

        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

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
        up_proj_activation = self.dense_h_to_4h(hidden_states)
        hidden_activation = self.act(up_proj_activation)
        output = self.dense_4h_to_h(hidden_activation)

        activations = MLPActivations(
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
        return self.dense_4h_to_h(self.act(self.dense_h_to_4h(hidden_states)))

class SurgicalGPTNeoXLayer(GPTNeoXLayer):
    def __init__(self, config: GPTNeoXConfig, layer_idx: int):
        super(GPTNeoXLayer, self).__init__()

        self.use_parallel_residual = config.use_parallel_residual
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        self.attention = SurgicalGPTNeoXAttention(config, layer_idx)
        self.mlp = SurgicalGPTNeoXMLP(config)

    def attn_unit_forward(
        self,
        hidden_states: th.FloatTensor,
        attention_mask: th.BoolTensor | None = None,
        position_embeddings: th.FloatTensor | None = None,
        **kwargs,
    ) -> th.FloatTensor:
        return self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def mlp_unit_forward(
        self,
        hidden_states: th.FloatTensor,
        **kwargs,
    ) -> th.FloatTensor:
        return self.mlp(self.post_attention_layernorm(hidden_states), **kwargs)

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

        attention_normed_input = self.input_layernorm(residual)
        attention_output = self.attention(
            hidden_states=attention_normed_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            activation_mask=activation_mask_for_attention,
        )
        if isinstance(attention_output, AttentionActivations):
            attention_activations = attention_output
        else:
            attention_activations = AttentionActivations(output=attention_output)

        attention_dropped_output = self.post_attention_dropout(attention_activations.output)
        activation_mask_for_mlp = [".".join(activation_path.split(".")[1:]) for activation_path in activation_mask if activation_path.startswith("mlp_activations.")]

        if not self.use_parallel_residual:
            residual += attention_dropped_output

        mlp_normed_input = self.post_attention_layernorm(residual)
        mlp_output = self.mlp(
            mlp_normed_input,
            activation_mask=activation_mask_for_mlp,
        )
        if isinstance(mlp_output, MLPActivations):
            mlp_activations = mlp_output
        else:
            mlp_activations = MLPActivations(output=mlp_output)

        mlp_dropped_output = self.post_mlp_dropout(mlp_activations.output)

        residual += mlp_dropped_output

        if self.use_parallel_residual:
            residual += attention_dropped_output

        activations = DecoderLayerActivations(
            attention_normed_input=attention_normed_input,
            attention_activations=attention_activations,
            attention_dropped_output=attention_dropped_output,
            mlp_normed_input=mlp_normed_input,
            mlp_activations=mlp_activations,
            mlp_dropped_output=mlp_dropped_output,
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

        attention_output = self.post_attention_dropout(
            self.attention(
                hidden_states=self.input_layernorm(residual),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                track_activations=False,
            )
        )

        if not self.use_parallel_residual:
            residual += attention_output

        mlp_output = self.post_mlp_dropout(
            self.mlp(
                self.post_attention_layernorm(residual),
                track_activations=False,
            )
        )

        residual += mlp_output

        if self.use_parallel_residual:
            residual += attention_output

        return residual

class SurgicalGPTNeoXRotaryEmbedding(GPTNeoXRotaryEmbedding):
    pass

class SurgicalGPTNeoXPreTrainedModel(GPTNeoXPreTrainedModel):
    pass

class SurgicalGPTNeoXModel(SurgicalGPTNeoXPreTrainedModel, GPTNeoXModel):
    def __init__(self, config: GPTNeoXConfig):
        SurgicalGPTNeoXPreTrainedModel.__init__(self, config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [SurgicalGPTNeoXLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = SurgicalGPTNeoXRotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def unit_forwards(self) -> list[Callable]:
        if self.config.use_parallel_residual:
            # eugh
            return [
                lambda hidden_states, position_embeddings, attention_mask, track_activations: layer.attn_unit_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    track_activations=track_activations,
                ) + layer.mlp_unit_forward(
                    hidden_states,
                    track_activations=track_activations,
                ) for layer in self.layers]

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
            inputs_embeds = self.embed_in(input_ids)

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
            inputs_embeds = self.embed_in(input_ids)

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

class SurgicalGPTNeoXForCausalLM(SurgicalGPTNeoXPreTrainedModel, GPTNeoXForCausalLM):
    def __init__(self, config: GPTNeoXConfig):
        SurgicalGPTNeoXPreTrainedModel.__init__(self, config)

        self.model = SurgicalGPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_in

    @classmethod
    def from_causal_lm(cls, model: GPTNeoXForCausalLM) -> "SurgicalGPTNeoXForCausalLM":
        with th.device("meta"):
            surgical_model = cls(model.config)

        surgical_model.embed_out = model.embed_out
        surgical_model.model.rotary_emb = model.gpt_neox.rotary_emb
        surgical_model.model.norm = model.gpt_neox.final_layer_norm
        for surgical_layer, model_layer in zip(surgical_model.model.layers, model.gpt_neox.layers):
            surgical_layer.mlp.dense_4h_to_h = model_layer.mlp.dense_4h_to_h
            surgical_layer.mlp.dense_h_to_4h = model_layer.mlp.dense_h_to_4h

            surgical_layer.attention.dense = model_layer.attention.dense
            surgical_layer.attention.query_key_value = model_layer.attention.query_key_value

            surgical_layer.post_mlp_dropout = model_layer.post_mlp_dropout
            surgical_layer.post_attention_dropout = model_layer.post_attention_dropout
            surgical_layer.post_attention_layernorm = model_layer.post_attention_layernorm
            surgical_layer.input_layernorm = model_layer.input_layernorm
            surgical_layer.use_parallel_residual = model_layer.use_parallel_residual
        surgical_model.model.emb_dropout = model.gpt_neox.emb_dropout
        surgical_model.model.embed_in = model.gpt_neox.embed_in

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

        logits = self.embed_out(model_activations.output)
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

        logits = self.embed_out(model_output)
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs) if labels is not None else None

        return loss
