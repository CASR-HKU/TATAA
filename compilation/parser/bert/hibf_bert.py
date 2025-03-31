"""
module for bert quantization, for now, only support for " bert-base-uncased " model on the text classification task (GLUE)
NOTE:  a total of 5 questions were left, please re-check before further used and released (Q1~Q5)
"""
import math
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from time import perf_counter


# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)


from parser.model_parser import ModelParser

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from quant_module import (
    HMQGeLU,
    HMQAct,
    HMQLinear,
    HMQMulQK,
    HMQSoftmax,
    HMQMulSV,
    HMQTanh,
    HMQLayerNorm,
    HMQConv2d,
)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # embedding layer in bert consist of 3 nn.embedding, which are not conv2D/linear-like, actually they operate like lookup tables.
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.quant = config.hibf_quant

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        if self.quant == True:
            self.LayerNorm = HMQLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        position_embedding_type=None,
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qact_sa_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.query = HMQLinear(
            in_features=config.hidden_size,
            out_features=self.all_head_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )
        self.key = HMQLinear(
            in_features=config.hidden_size,
            out_features=self.all_head_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )
        self.value = HMQLinear(
            in_features=config.hidden_size,
            out_features=self.all_head_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        self.mul_qk = HMQMulQK(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        self.sftm = HMQSoftmax()
    
        self.mul_sv = HMQMulSV(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        i: int,
        hidden_states: torch.Tensor,
        mp: ModelParser,
        C_dict=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        
        hidden_states = self.qact_sa_input(hidden_states)
        mixed_query_layer = self.query(hidden_states, self.qact_sa_input.quantizer, mp, 'block.'+str(i)+'.'+'query')
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(
            self.key(hidden_states, self.qact_sa_input.quantizer, mp, 'block.'+str(i)+'.'+'key')
        )
        value_layer = self.transpose_for_scores(
            self.value(hidden_states, self.qact_sa_input.quantizer, mp, 'block.'+str(i)+'.'+'value')
        )
            
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = self.mul_qk(query_layer, key_layer, 'block.'+str(i)+'.'+'mul_qk', mp)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # mp.parse_ops('mul', in_data=attention_scores, in_data_name='A0', in_tmp=torch.tensor([]), in_tmp_name='C0', out_data=attention_scores, out_data_name='A0')
        C_dict['C0'] = 1 / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            # mp.parse_ops('add', in_data=attention_scores, in_data_name='A0', in_tmp=attention_mask, in_tmp_name='B0', out_data=attention_scores, out_data_name='A0')

        # Normalize the attention scores to probabilities.
        attention_probs = self.sftm(attention_scores, 'block.'+str(i)+'.'+'softmax', mp, C_dict)

        context_layer = self.mul_sv(attention_probs, value_layer, 'block.'+str(i)+'.'+'mul_sv', mp)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg

        self.qact_so_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.dense = HMQLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )
        self.LayerNorm = HMQLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, i: int, hidden_states: torch.Tensor, input_tensor: torch.Tensor, mp: ModelParser, C_dict=None,
    ) -> torch.Tensor:
        
        hidden_states = self.qact_so_input(hidden_states)
        hidden_states = self.dense(hidden_states, self.qact_so_input.quantizer, mp, 'block.'+str(i)+'.'+'self_output', quant_type='fp16')
        hidden_states = self.LayerNorm((hidden_states.bfloat16() + input_tensor.bfloat16()).bfloat16(), 'block.'+str(i)+'.'+'ln_self_output', mp, C_dict)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(
        self,
        config,
        position_embedding_type=None,
    ):
        super().__init__()

        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type,)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        i,
        hidden_states,
        mp: ModelParser,
        C_dict=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            i,
            hidden_states,
            mp,
            C_dict,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(i, self_outputs[0], hidden_states, mp, C_dict)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg

        self.qact_itmd_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.dense = HMQLinear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        self.intermediate_act_fn = HMQGeLU()

    def forward(self, i:int, hidden_states: torch.Tensor, mp: ModelParser, C_dict=None) -> torch.Tensor:
        hidden_states = self.qact_itmd_input(hidden_states)
        hidden_states = self.dense(hidden_states, self.qact_itmd_input.quantizer, mp, layer_name='block.'+str(i)+'.'+'fc1', quant_type='fp16')  # fc1
        hidden_states = self.intermediate_act_fn(hidden_states, 'block.'+str(i)+'.'+'gelu', mp, C_dict)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg

        self.qact_fno_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.dense = HMQLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )
        self.LayerNorm = HMQLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, i:int, hidden_states: torch.Tensor, input_tensor: torch.Tensor, mp: ModelParser, C_dict=None
    ) -> torch.Tensor:
        hidden_states = self.qact_fno_input(hidden_states)

        hidden_states = self.dense(hidden_states, self.qact_fno_input.quantizer, mp, layer_name='block.'+str(i)+'.'+'fc2', quant_type='fp16')  # fc2
        hidden_states = self.LayerNorm((hidden_states.bfloat16() + input_tensor.bfloat16()).bfloat16(), 'block.'+str(i)+'.'+'ln_out', mp, C_dict)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        i: int,
        hidden_states: torch.Tensor,
        mp: ModelParser,
        C_dict=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            i,
            hidden_states,
            mp,
            C_dict,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights 
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
            i,
            mp,
            C_dict
        )
        outputs = (layer_output,) + outputs

        return outputs

    # FFN
    def feed_forward_chunk(self, attention_output, i:int, mp:ModelParser, C_dict=None):
        intermediate_output = self.intermediate(i, attention_output, mp, C_dict)
        layer_output = self.output(i, intermediate_output, attention_output, mp, C_dict)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            self.layer.append(BertLayer(config))

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        in_shape = hidden_states.shape
        batch_size = in_shape[0]
        model_name = 'bert_base_uncased'
        mp = ModelParser(model_name, batch_size)
        C_dict = {}

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            print('self.layer')
            # start.record()
            start_time = perf_counter()

            layer_outputs = layer_module(
                i,
                hidden_states,
                mp,
                C_dict,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            # end.record()
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end))
            print("Power: ", torch.cuda.power_draw())
            torch.cuda.synchronize()
            end_time = perf_counter()
            print(end_time - start_time)
            
            break
            
        

            hidden_states = layer_outputs[0]
        print('self.layer done')

        ms = mp.return_ms()    
        ms.update_param('constants', C_dict)
        ms.save(f"./model_spec/{model_name}_bs{batch_size}.json")

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg

        self.qact_pooler_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.dense = HMQLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        if config.tanh_quant:
            if config.hibf_quant == True:
                self.activation = HMQTanh()  #### Padé Approximation for Tanh
            else:
                self.activation = nn.Tanh()
        else:
            self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]

        first_token_tensor = self.qact_pooler_input(first_token_tensor)

        pooled_output = self.dense(first_token_tensor, self.qact_pooler_input.quantizer)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    _no_split_modules = ["BertEmbeddings"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # [b, L]
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        print("BertModel Forward")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        print("device: ", device)

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        use_sdpa_attention_masks = False  # normal attention
        # Expand the attention mask
        if use_sdpa_attention_masks:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape
            )
            # extended_attention_mask = None
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        head_mask = None

        print("self.encoder")
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print("self.encoder done")
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class QBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.quant = config.hibf_quant
        self.calibrate = config.calibrate
        self.quant_cfg = config.quant_cfg

        self.bert = BertModel(config)

        self.qact_cla_input = HMQAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_A,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_A,
            observer_str=self.quant_cfg.OBSERVER_A,
            quantizer_str=self.quant_cfg.QUANTIZER_A,
        )

        self.classifier = HMQLinear(
            in_features=config.hidden_size,
            out_features=config.num_labels,
            bias=True,
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=self.quant_cfg.BIT_TYPE_W,
            calibration_mode=self.quant_cfg.CALIBRATION_MODE_W,
            observer_str=self.quant_cfg.OBSERVER_W,
            quantizer_str=self.quant_cfg.QUANTIZER_W,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def model_quant(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.quant = True

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [HMQConv2d, HMQLinear, HMQAct, HMQMulQK, HMQMulSV]:
                m.calibrate = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        print("self.bert")
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print("self.bert done")

        pooled_output = outputs[1]
        pooled_output = self.qact_cla_input(pooled_output)
        logits = self.classifier(pooled_output, self.qact_cla_input.quantizer)
        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
