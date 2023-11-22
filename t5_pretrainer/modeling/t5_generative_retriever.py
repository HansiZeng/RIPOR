import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union, List, Iterable, Callable
from dataclasses import dataclass
import inspect

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import numpy as np
from transformers import T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Config, T5Stack
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.utils import logging
from transformers.models.t5.configuration_t5 import T5Config
import transformers
import torch.nn as nn 
import torch
import copy
import warnings

from .customized_modeling_t5 import DecoderT5Stack

logger = logging.get_logger(__name__)

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

EPSILON = np.finfo(np.float32).tiny


@dataclass
class T5GenRetModelOutput(Seq2SeqLMOutput):
    decoder_last_hidden_state: torch.FloatTensor = None
    list_decoder_embeds: Tuple[torch.FloatTensor] = None

class T5forDocIDConfig(T5Config):
    def __init__(self, 
                 decoder_vocab_sizes= [256 for _ in range(32)],
                    # 5_000, 5_000, 5_000, 5_000, 5_000, 5_000, 5_000, 5_000
                    #[50_000, 50_000, 50_000, 50_000],  #[100_000, 100_000], 
                 decoding=False,
                 decoder_start_token_path="./t5_decoder_start_token_embeds/t5-base.npy",
                 apply_decoder_t5_stack=False,
                 scaleup_output_hidden=False,
                 shared_output_input_embeds=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_vocab_sizes = decoder_vocab_sizes
        self.decoder_start_token_path = decoder_start_token_path
        self.tie_word_embeddings = False
        self.decoding = decoding
        self.max_decoder_length = len(decoder_vocab_sizes)
        self.apply_decoder_t5_stack=apply_decoder_t5_stack
        self.scaleup_output_hidden=scaleup_output_hidden
        self.shared_output_input_embeds = shared_output_input_embeds

        assert self.apply_decoder_t5_stack == False
    
# base model
class T5ForDocIDGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5forDocIDConfig):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        if config.apply_decoder_t5_stack:
            self.decoder = DecoderT5Stack(decoder_config, embed_tokens=None)
        else:
            self.decoder = T5Stack(decoder_config, embed_tokens=None)

        # create embeds
        self.list_decoder_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, config.d_model) for vocab_size in config.decoder_vocab_sizes
        ])
        if not config.shared_output_input_embeds:
            self.list_output_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, config.d_model) for vocab_size in config.decoder_vocab_sizes
        ]) 
            
        self.num_decoder_embeds = len(self.list_decoder_embeds)
        self.start_token_embed = nn.Parameter(torch.randn(1, 1, config.d_model))
    
        # Initialize weights and apply final processing
        self.post_init()
        if config.num_decoder_layers == 12 and config.num_heads == 12:
            #print("this is the t5-base model")
            assert "t5-base" in config.decoder_start_token_path, config.decoder_start_token_path
            np_embed = torch.from_numpy(np.load(config.decoder_start_token_path)).float()
            assert np_embed.size() == self.start_token_embed.data.size()
            self.start_token_embed.data = np_embed
        elif config.num_decoder_layers == 24 and config.num_heads == 16:
            print("this is the t5-large model")
            assert "t5-large" in config.decoder_start_token_path, config.decoder_start_token_path
            np_embed = torch.from_numpy(np.load(config.decoder_start_token_path)).float()
            assert np_embed.size() == self.start_token_embed.data.size()
            self.start_token_embed.data = np_embed
        elif config.num_decoder_layers == 24 and config.num_heads == 32:
            print("this is the t5-3b model")
            assert "t5-3b" in config.decoder_start_token_path, config.decoder_start_token_path
            np_embed = torch.from_numpy(np.load(config.decoder_start_token_path)).float()
            assert np_embed.size() == self.start_token_embed.data.size()
            self.start_token_embed.data = np_embed
        else:
            raise ValueError("the model with decoer layers {} is not supported.".format(config.num_decoder_layers))

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return None 

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_decoder_inputs_embeds(self, decoder_input_ids: Optional[torch.LongTensor]):
        """
        case 1: [[-1], [-1]]
        case 2: [[-1, 24], [-1, 0]]
        """
        input_shape = decoder_input_ids.size() 
        batch_size, seq_length = input_shape
        assert seq_length <= self.num_decoder_embeds

        start_token_embeds = self.start_token_embed.expand(batch_size, 1, -1) #[bz, 1, d_model]
        assert decoder_input_ids[0,0] == -1 or decoder_input_ids[0,0] == 0 
        if seq_length == 1:
            input_embeds = start_token_embeds
        else:
            input_embeds = [start_token_embeds]
            for i in range(1, seq_length):
                embeds = self.list_decoder_embeds[i-1](decoder_input_ids[:, i]).unsqueeze(1) #[bz, 1, d_model]
                input_embeds.append(embeds)
            input_embeds = torch.cat(input_embeds, dim=1)

        return input_embeds

    def get_decoder_mul_inputs_embeds(self, decoder_input_ids: Optional[torch.LongTensor]):
        """
        Args:
            [bz, seq_len, mul_size]
            case 1: [
                    [[-1,-1]],
                    [[-1,-1]]
                    ]
            case 2: [
                [[-1, -1], [2, 4]],
                [[-1, -1], [3, 1]],
                    ]
        Returns:
            [bz, seq_len, d_model]
        """
        assert decoder_input_ids.dim() == 3
        bz, seq_length, mul_size = decoder_input_ids.size()
        assert seq_length <= self.num_decoder_embeds

        start_token_embeds = self.start_token_embed.expand(bz, 1, -1) #[bz, 1, d_model]
        assert decoder_input_ids[0,0,0] == -1 or decoder_input_ids[0,0,0] == 0 
        if seq_length == 1:
            input_embeds = start_token_embeds
        else:
            input_embeds = [start_token_embeds]
            for i in range(1, seq_length):
                embed_layer = self.list_decoder_embeds[i-1]
                step_input_ids = decoder_input_ids[:, i, :].clone() #[bz, mul_size]
                embeds = embed_layer[step_input_ids].mean(dim=1).unsqueeze(1) # [bz, mul_size, d_model] --> [bz, 1, d_model]
                input_embeds.append(embeds)
            input_embeds = torch.cat(input_embeds, dim=1)

        return input_embeds

    def get_lm_logits(self, sequence_output):
        # sequence_output: [bz, seq_length, d_model]
        # lm_logits: list of [bz, vocab_size], the list length is max_seq_length
        lm_logits = []
        for i in range(sequence_output.size(1)):
            if self.config.shared_output_input_embeds:
                logits = (sequence_output[:, i, :] @ self.list_decoder_embeds[i].weight.t())
            else:
                logits = (sequence_output[:, i, :] @ self.list_output_embeds[i].weight.t())
            
            lm_logits.append(logits)

        return lm_logits
    
    def assign_decoder_embed(self, path, idx):
        embed = torch.from_numpy(np.load(path)).float()
        assigned_embed = self.list_decoder_embeds[idx].weight.data 
        #assert embed.size() == assigned_embed.size(), (embed.size(), assigned_embed.size())
        assert embed.dtype == assigned_embed.dtype, (embed.dtype, assigned_embed.dtype)

        if embed.size() == assigned_embed.size():
            self.list_decoder_embeds[idx].weight.data = copy.deepcopy(embed) 
        elif embed.size()[0] < assigned_embed.size()[0]:
            embed.size()[1] == assigned_embed.size()[1], (embed.size()[1], assigned_embed.size()[1])
            self.list_decoder_embeds[idx].weight.data[:embed.size()[0], :] = copy.deepcopy(embed) 
        else:
            raise ValueError("embed's shape: {} should no larger than assigned_embed'shape: {}".format(
                embed.size(), assigned_embed.szie()
            ))
        if not self.config.shared_output_input_embeds:
            self.list_output_embeds[idx].weight.data = copy.deepcopy(embed)

    def resize_and_assign_decoder_embed(self, path, idx):
        if isinstance(path, str):
            embed = torch.from_numpy(np.load(path)).float()
        elif isinstance(path, torch.Tensor):
            embed = path
        else:
            raise ValueError(f"the {path} doesn't have valid type.")
        print("the shape of new assigned embed: ", embed.shape)
        self.list_decoder_embeds[idx].weight.data = copy.deepcopy(embed) 
        if not self.config.shared_output_input_embeds:
            self.list_output_embeds[idx].weight.data = copy.deepcopy(embed)
        self.config.decoder_vocab_sizes[idx] = len(embed)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_list_decoder_embeds: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            #decoder_input_ids = self._shift_right(labels)
            raise NotImplementedError

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        if decoder_input_ids.dim() == 2:
            decoder_inputs_embeds = self.get_decoder_inputs_embeds(decoder_input_ids)
        elif decoder_input_ids.dim() == 3:
            decoder_inputs_embeds = self.get_decoder_mul_inputs_embeds(decoder_input_ids)
        else:
            raise ValueError(f"decoder_input_ids: {decoder_input_ids.size()} shape is not valid")
        
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        #if self.config.tie_word_embeddings: # we remove this anyway
        if self.config.scaleup_output_hidden:
            sequence_output = sequence_output * (self.model_dim**-0.5) #[bz, seq_length, d_model]
        assert sequence_output.size(1) == decoder_input_ids.size(1), (sequence_output.size(1), decoder_input_ids.size(1))

        lm_logits = None
        if self.config.decoding:
            lm_logits = self.get_lm_logits(sequence_output)

        if not return_dict:
            raise NotImplementedError

        return T5GenRetModelOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_last_hidden_state=sequence_output,
            list_decoder_embeds=tuple(self.list_decoder_embeds) if return_list_decoder_embeds else None
        )
              
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

# t5_dense_encoder
class T5DocIDGenEncoder(nn.Module):
    def __init__(self, model_name_or_path, model_args=None, ):
        super().__init__()
        config = T5forDocIDConfig().from_pretrained(model_name_or_path)
        config.decoding = False

        self.base_model = T5ForDocIDGeneration.from_pretrained(model_name_or_path, config=config)
        self.config = config
        self.model_args = model_args

    def forward(self, **inputs):
        return self.encode(**inputs)

    def doc_encode(self, **inputs):
        return self.encode(**inputs)
    
    def query_encode(self, **inputs):
        return self.encode(**inputs)
    
    def encode(self, input_ids, attention_mask, decoder_input_ids, return_decoder_last_hidden_state=False):
        assert decoder_input_ids.size(1) <= self.config.max_decoder_length
        idx = decoder_input_ids.size(1) - 1 
        model_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        reps = model_output.decoder_last_hidden_state[:, idx, :] #[bz, d_model]

        if return_decoder_last_hidden_state:
            return reps, model_output.decoder_last_hidden_state
        else:
            return reps
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path=None, model_args=None):
        return cls(model_name_or_path, model_args)

    def assign_decoder_embed(self, path, idx):
        self.base_model.assign_decoder_embed(path, idx)

    def resize_and_assign_decoder_embed(self, path, idx):
        self.base_model.resize_and_assign_decoder_embed(path, idx)

class T5SeqPretrainEncoder(T5DocIDGenEncoder):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        
        self.rank_loss = nn.MSELoss()
        self.commit_loss = nn.CrossEntropyLoss()

    def doc_encode(self, **inputs):
        text_reps = self.base_model(**inputs).decoder_last_hidden_state
        last_idx = text_reps.size(1) - 1
        rep = text_reps[:, last_idx, :]

        return rep

    def query_encode(self, **inputs):
        return self.doc_encode(**inputs)

    def get_rank_loss(self, pq_last_rep, nq_last_rep,
                      pdoc_last_rep, ndoc_last_rep,
                      teacher_pos_score, teacher_neg_score,
                      pq_prev_reps=None, nq_prev_reps=None, 
                      pos_prev_smtids=None, neg_prev_smtids=None):
        if pq_prev_reps is None:
            assert nq_prev_reps is None and pos_prev_smtids is None and neg_prev_smtids is None
        else:
            raise NotImplementedError
            bz, prev_len = pos_prev_smtids.size()
            pq_prev_reps = pq_prev_reps.view(bz*prev_len, -1)
            nq_prev_reps = nq_prev_reps.view(bz*prev_len, -1)

            pdoc_pvcen_reps = []
            ndoc_pvcen_reps = []
            for i in range(prev_len):
                embed_layer = self.base_model.list_decoder_embeds[i]
                pdoc_pvcen_reps.append(embed_layer(pos_prev_smtids[:, i].clone()).unsqueeze(1))
                ndoc_pvcen_reps.append(embed_layer(neg_prev_smtids[:, i].clone()).unsqueeze(1))
            pdoc_pvcen_reps = torch.cat(pdoc_pvcen_reps, dim=1).view(bz*prev_len, -1)
            ndoc_pvcen_reps = torch.cat(ndoc_pvcen_reps, dim=1).view(bz*prev_len, -1)

            with torch.no_grad():
                pos_prev_logit = (pq_prev_reps * pdoc_pvcen_reps).sum(dim=-1).view(bz, prev_len).sum(dim=-1) #[bz]
                neg_prev_logit = (nq_prev_reps * ndoc_pvcen_reps).sum(dim=-1).view(bz, prev_len).sum(dim=-1) #[bz]
        
        pos_last_logit = (pq_last_rep * pdoc_last_rep).sum(dim=-1)
        neg_last_logit = (nq_last_rep * ndoc_last_rep).sum(dim=-1)
        if pq_prev_reps is None:
            margin = pos_last_logit - neg_last_logit
        else:
            raise NotImplementedError
            margin = (pos_prev_logit + pos_last_logit) - (neg_prev_logit + neg_last_logit)
        
        if teacher_pos_score is not None and teacher_neg_score is not None:
            teacher_margin = teacher_pos_score - teacher_neg_score
            loss = self.rank_loss(margin, teacher_margin)
        else:
            loss = torch.log(1 + torch.exp(-margin)).mean()

        return loss 

    def get_commit_loss(self,
                        pq_prev_reps,
                        pd_prev_reps,
                        nd_prev_reps,
                        pos_prev_smtids,
                        neg_prev_smtids):
        """
        reps: [bz, smtid_len-1, d_model]
        smtids: [bz, smtid_len-1] or [bz, smtid_len-1, mul_size]
        """
        assert pos_prev_smtids.dim() == neg_prev_smtids.dim()
        hard_label = True
        if pos_prev_smtids.dim() == 2:
            pos_doc_labels = pos_prev_smtids.clone()
            neg_doc_labels = neg_prev_smtids.clone()
            pos_query_labels = pos_doc_labels.clone()
        elif pos_prev_smtids.dim() == 3:
            hard_label = False
            vocab_size = self.base_model.list_decoder_embeds[0].weight.size(0)
            bz, prev_len, mul_size = pos_prev_smtids.size()
            
            pos_prev_smtids = pos_prev_smtids.view(bz*prev_len, mul_size)
            pos_doc_labels = torch.zeros((bz*prev_len, vocab_size)).to(pos_prev_smtids.device)
            pos_doc_labels = pos_doc_labels.scatter_(1, pos_prev_smtids, 1).view(bz, prev_len, mul_size)

            neg_prev_smtids = neg_prev_smtids.view(bz*prev_len, mul_size)
            neg_doc_labels = torch.zeros((bz*prev_len, vocab_size)).to(neg_prev_smtids)
            neg_doc_labels = neg_doc_labels.scatter_(1, neg_prev_smtids, 1).view(bz, prev_len, vocab_size)

            pos_query_labels = pos_doc_labels.clone()        
        else:
            raise ValueError("pos_doc_labels with size = {} is not valid".format(pos_prev_smtids.dim()))
        
        all_labels = [pos_doc_labels, neg_doc_labels, pos_query_labels]
        all_reps = [pd_prev_reps, nd_prev_reps, pq_prev_reps]

        loss = 0.
        bz, prev_len, _ = pq_prev_reps.size()
        for reps, labels in zip(all_reps, all_labels):
            logits = []
            for i in range(reps.size(1)):
                embed_weight = self.base_model.list_decoder_embeds[i].weight
                logits.append((reps[:, i].clone() @ embed_weight.t()).view(bz, 1, -1))
            logits = torch.cat(logits, dim=1).view(bz*prev_len, -1)

            if hard_label:
                labels = labels.view(bz*prev_len)
            else:
                labels = labels.view(bz*prev_len, vocab_size)
                print("soft_label_size: ", (labels != 0.).sum(dim=-1).view(-1, 1))
                labels = labels / (labels != 0.).sum(dim=-1).view(-1, 1)
            loss += self.commit_loss(logits, labels)

        return loss 

    def cond_prev_smtid_query_doc_score(self, **inputs):
        """
        tokenized_query: [bz, seq_length]
        tokenized_doc: [bz, seq_length]

        prev_smtids: [bz, smtid_length-1]
        """
        query_reps = self.base_model(**inputs["tokenized_query"]).decoder_last_hidden_state
        doc_reps = self.base_model(**inputs["tokenized_doc"]).decoder_last_hidden_state

        bz, smtid_len, _ = query_reps.size()
        
        last_idx = smtid_len - 1
        prev_len = smtid_len - 1
        query_last_rep = query_reps[:, last_idx, :] #[bz, d_model]
        doc_last_rep = doc_reps[:, last_idx, :] #[bz, d_model]

        if last_idx == 0:
            last_logit = torch.sum(query_last_rep * doc_last_rep, dim=-1) #[bz]
            return last_logit
        else:
            prev_smtids = inputs["prev_smtids"]
            assert prev_len == prev_smtids.size(1), (prev_len, prev_smtids)

            doc_pvcen_reps = []
            for i in range(prev_len):
                embed_layer = self.base_model.list_decoder_embeds[i]
                doc_pvcen_reps.append(embed_layer(prev_smtids[:, i].clone()).unsqueeze(1)) #[bz, 1, d_model]
            doc_pvcen_reps = torch.cat(doc_pvcen_reps, dim=1).view(bz*prev_len, -1)
            query_pv_reps = query_reps[:, :last_idx, :].clone() #[bz, prev_len, d_model]

            pv_logit = torch.sum(query_pv_reps.view(bz*prev_len, -1) * doc_pvcen_reps, dim=-1).view(bz, prev_len).sum(dim=-1) #[bz]
            last_logit = torch.sum(query_last_rep * doc_last_rep, dim=-1) #[bz]

            return pv_logit + last_logit        

    def forward(self, **inputs):
        """
        tokenized_pos_query: [bz, seq_length]
        tokenized_neg_query: [bz, seq_length]
        tokenized_pos_doc: [bz, seq_length]
        tokenized_neg_doc: [bz, seq_length]

        Optional:
        pos_prev_smtids: [bz, smtid_length-1]
        neg_prev_smtids: [bz, smtid_length-1]
        """
        pos_query_reps = self.base_model(**inputs["tokenized_pos_query"]).decoder_last_hidden_state # [bz, smtid_length, d_model]
        neg_query_reps = self.base_model(**inputs["tokenized_neg_query"]).decoder_last_hidden_state
        pos_doc_reps = self.base_model(**inputs["tokenized_pos_doc"]).decoder_last_hidden_state
        neg_doc_reps = self.base_model(**inputs["tokenized_neg_doc"]).decoder_last_hidden_state

        last_idx = pos_query_reps.size(1) - 1

        # check
        if pos_query_reps.size(1) == 1:
            "pos_prev_smtids" not in inputs and "neg_prev_smtids" not in inputs 
        else:
            "pos_prev_smtids" in inputs and "neg_prev_smtids" in inputs
        
        pq_last_rep = pos_query_reps[:, last_idx, :].clone()
        nq_last_rep = neg_query_reps[:, last_idx, :].clone()
        
        pdoc_last_rep = pos_doc_reps[:, last_idx, :].clone()
        ndoc_last_rep = neg_doc_reps[:, last_idx, :].clone()

        if pos_query_reps.size(1) == 1:
            rank_loss = self.get_rank_loss(pq_last_rep=pq_last_rep, nq_last_rep=nq_last_rep,
                                           pdoc_last_rep=pdoc_last_rep, ndoc_last_rep=ndoc_last_rep,
                                           teacher_pos_score=inputs["teacher_pos_score"],
                                            teacher_neg_score=inputs["teacher_neg_score"])
        else:
            pq_prev_reps = pos_query_reps[:, :last_idx, :].clone()
            nq_prev_reps = None #neg_query_reps[:, :last_idx, :].clone()
            rank_loss = self.get_rank_loss(pq_last_rep=pq_last_rep, nq_last_rep=nq_last_rep,
                                           pdoc_last_rep=pdoc_last_rep, ndoc_last_rep=ndoc_last_rep,
                                           teacher_pos_score=inputs["teacher_pos_score"],
                                            teacher_neg_score=inputs["teacher_neg_score"])
                                            #pq_prev_reps=pq_prev_reps, nq_prev_reps=nq_prev_reps,
                                            #pos_prev_smtids=inputs["pos_prev_smtids"], neg_prev_smtids=inputs["neg_prev_smtids"])
        

        if pos_query_reps.size(1) == 1:
            return {
                "rank": rank_loss
            }
        else:
            pd_prev_reps = pos_doc_reps[:, :last_idx, :].clone()
            nd_prev_reps = neg_doc_reps[:, :last_idx, :].clone()
            commit_loss = self.get_commit_loss(pq_prev_reps=pq_prev_reps,
                                            pd_prev_reps=pd_prev_reps,
                                            nd_prev_reps=nd_prev_reps,
                                            pos_prev_smtids=inputs["pos_prev_smtids"].clone(),
                                            neg_prev_smtids=inputs["neg_prev_smtids"].clone())
            return {
                "commit": commit_loss,
                "rank": rank_loss
            }

# T5Seq
class T5SeqAQEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, shared_output_input_embeds, multi_vocab_sizes=None):
        super().__init__()
        config = T5forDocIDConfig().from_pretrained(model_name_or_path)
        config.decoding = False

        if shared_output_input_embeds is not None:
            assert shared_output_input_embeds in [False, True]
            config.shared_output_input_embeds = shared_output_input_embeds

        self.base_model = T5ForDocIDGeneration.from_pretrained(model_name_or_path, config=config)
        self.config = config
        self.model_args = None # incompatible with previous models
            
    def query_encode(self, **inputs):
        text_reps = self.base_model(**inputs).decoder_last_hidden_state
        last_idx = text_reps.size(1) - 1
        assert last_idx == 0
        rep = text_reps[:, last_idx, :]

        return rep
    
    def rerank_forward(self, **inputs):
        query_embeds = self.base_model(**inputs["tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        doc_embeds = self.decode(inputs["doc_encoding"]) #[bz, smtid_length, d_model]

        return (query_embeds * doc_embeds).sum(dim=-1).sum(dim=-1)

    def rerank_forward_for_sanity_check(self, **inputs):
        """
        Args:
            query_embeds: [bz, seq_length] + [bz, smtid_length]
            doc_encodings: [bz, smtid_length]
        """
        query_embeds = self.base_model(**inputs["tokenized_query"]).decoder_last_hidden_state[:, 0, :].unsqueeze(1) #[bz, 1, d_model]
        doc_embeds = self.decode(inputs["doc_encodings"]) #[bz, smtid_length,d_model]

        return (query_embeds * doc_embeds).sum(dim=-1).sum(dim=-1)

    def decode(self, text_encodings, summation=False):
        """
        Args:
            text_encodings: [bz, smtid_length]
        Returns:
            text_embeds: [bz, smtid_length, d_model] or [bz, d_model] if summation=True
        """
        text_embeds = []
        bz, smtid_length = text_encodings.size()
        for i in range(smtid_length):
            if self.config.shared_output_input_embeds:
                text_embeds.append(self.base_model.list_decoder_embeds[i](text_encodings[:, i].clone()).unsqueeze(1))
            else:
                text_embeds.append(self.base_model.list_output_embeds[i](text_encodings[:, i].clone()).unsqueeze(1))
        text_embeds = torch.cat(text_embeds, dim=1) #[bz, smtid_length, d_model]

        if summation:
            return text_embeds.sum(dim=1)

        return text_embeds
                       
    def assign_output_embeds(self, code_embeddings):
        if self.config.shared_output_input_embeds:
            assert len(code_embeddings) == len(self.base_model.list_decoder_embeds)
        else:
            assert len(code_embeddings) == len(self.base_model.list_output_embeds)

        for i in range(len(code_embeddings)):
            code_embed = copy.deepcopy(torch.from_numpy(code_embeddings[i]))
            assert code_embed.size() == self.base_model.list_output_embeds[i].weight.size()
            if self.config.shared_output_input_embeds:
                self.base_model.list_decoder_embeds[i].weight.data = code_embed
                if i == 0:
                    print("only decoder embed is assigned centroid embeds.")
            else:
                self.base_model.list_output_embeds[i].weight.data = code_embed 
                self.base_model.list_decoder_embeds[i].weight.data = copy.deepcopy(code_embed)
                if i == 0:
                    print("both decoder and output embed are assigned centroid embeds")
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path=None, shared_output_input_embeds=None, multi_vocab_sizes=False):
        return cls(model_name_or_path, shared_output_input_embeds, multi_vocab_sizes)

class T5SeqAQEncoderForMarginMSE(T5SeqAQEncoder):
    def __init__(self, model_name_or_path, shared_output_input_embeds, multi_vocab_sizes=None):
        super().__init__(model_name_or_path, shared_output_input_embeds)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        """
        Args:
            pos_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            neg_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            pos_doc_encoding: [bz, smtid_length]
            neg_doc_encoding: [bz, smtid_length]

            teacher_pos_scores: [bz]
            teacher_neg_scores: [bz]
        """
        pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
        neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_Model]

        student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        loss = self.loss_fn(student_margin, teacher_margin)

        return {"rank": loss}

class T5AQEncoder(T5SeqAQEncoder):
    """ This is for proof of concept """
    def __init__(self, model_name_or_path, shared_output_input_embeds, multi_vocab_sizes=None):
        super().__init__(model_name_or_path, shared_output_input_embeds)

    def query_encode(self, **inputs):
        text_reps = self.base_model(**inputs).decoder_last_hidden_state
        last_idx = text_reps.size(1) - 1
        assert last_idx == 0
        rep = text_reps[:, last_idx, :]

        return rep

    def decode(self, text_encodings):
        return super().decode(text_encodings, summation=True)

class T5SeqAQEncoderForLngKnpMarginMSE(T5SeqAQEncoder):
    def __init__(self, model_name_or_path, shared_output_input_embeds, multi_vocab_sizes=None):
        super().__init__(model_name_or_path, shared_output_input_embeds)

        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, **inputs):
        """
        Args:
            pos_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            neg_tokenized_query: [bz, seq_length] + [bz, smtid_length]
            pos_doc_encoding: [bz, smtid_length]
            neg_doc_encoding: [bz, smtid_length]

            teacher_pos_scores: [bz]
            teacher_neg_scores: [bz]

            smtid_8_teacher_pos_scores: [bz]
            smtid_8_teacher_neg_scores: [bz]
            smtid_16_teacher_pos_scores: [bz]
            smtid_16_teacher_neg_scores: [bz]
        """

        pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
        neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_Model]
        
        # rank loss
        student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]
        rank_loss = self.loss_fn(student_margin, teacher_margin)

        # rank_4 loss 
        early_pos_score = (pos_query_embeds[:, :4, :].clone() * pos_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_neg_score = (neg_query_embeds[:, :4, :].clone() * neg_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
        early_margin = early_pos_score - early_neg_score
        early_teacher_margin = inputs["smtid_4_teacher_pos_scores"] - inputs["smtid_4_teacher_neg_scores"]
        rank_4_loss = self.loss_fn(early_margin, early_teacher_margin)

        if pos_doc_embeds.size()[1] == 8:
            return {"rank": rank_loss, "rank_4": rank_4_loss}
        elif pos_doc_embeds.size()[1] in [16, 32]:
            # rank_8 loss 
            early_pos_score = (pos_query_embeds[:, :8, :].clone() * pos_doc_embeds[:, :8, :].clone()).sum(-1).sum(-1)
            early_neg_score = (neg_query_embeds[:, :8, :].clone() * neg_doc_embeds[:, :8, :].clone()).sum(-1).sum(-1)
            early_margin = early_pos_score - early_neg_score
            early_teacher_margin = inputs["smtid_8_teacher_pos_scores"] - inputs["smtid_8_teacher_neg_scores"]
            rank_8_loss = self.loss_fn(early_margin, early_teacher_margin)

            if pos_doc_embeds.size()[1] == 16:
                return {"rank": rank_loss, "rank_8": rank_8_loss, "rank_4": rank_4_loss}
            elif pos_doc_embeds.size()[1] == 32:
                # rank_16 loss
                early_pos_score = (pos_query_embeds[:, :16, :].clone() * pos_doc_embeds[:, :16, :].clone()).sum(-1).sum(-1)
                early_neg_score = (neg_query_embeds[:, :16, :].clone() * neg_doc_embeds[:, :16, :].clone()).sum(-1).sum(-1)
                early_margin = early_pos_score - early_neg_score
                early_teacher_margin = inputs["smtid_16_teacher_pos_scores"] - inputs["smtid_16_teacher_neg_scores"]
                rank_16_loss = self.loss_fn(early_margin, early_teacher_margin)
                
                return {"rank": rank_loss, "rank_4": rank_4_loss, "rank_8": rank_8_loss, "rank_16": rank_16_loss}
            else:
                raise ValueError("not valid length: {}".format(pos_doc_embeds.size()[1]))
        else:
            raise ValueError("not valid length: {}".format(pos_doc_embeds.size()[1]))

class T5SeqAQEncoderForSeq2Seq(T5SeqAQEncoder):
    def __init__(self, model_name_or_path, shared_output_input_embeds, multi_vocab_sizes=False):
        super().__init__(model_name_or_path, shared_output_input_embeds, multi_vocab_sizes)
        self.rank_loss = nn.CrossEntropyLoss()
        self.multi_vocab_sizes = multi_vocab_sizes
        if self.multi_vocab_sizes:
            print("seq2seq model has multi_vocab_sizes")

    def get_seq_logits(self, text_embeds):
        """
        text_embeds: [bz, smtid_length, d_model]
        """
        bz, smtid_length, d_model = text_embeds.size()
        seq_logits = []
        for i in range(smtid_length):
            if self.config.shared_output_input_embeds:
                embed_weight = self.base_model.list_decoder_embeds[i].weight
                #raise NotImplementedError
            else:
                embed_weight = self.base_model.list_output_embeds[i].weight
            seq_logits.append(
                (text_embeds[:, i].clone() @ embed_weight.t()).view(bz, 1, -1)
                )
        
        if self.multi_vocab_sizes:
            return seq_logits # list of [bz, vocab_size]
        else: 
            seq_logits = torch.cat(seq_logits, dim=1) # [bz, smtid_length, vocab_size]
            return seq_logits


    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """
        query_embeds = self.base_model(**inputs["tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]
        logits = self.get_seq_logits(query_embeds)

        bz, smtid_length = inputs["labels"].size()
        if self.multi_vocab_sizes:
            loss = 0.
            for i in range(smtid_length):
                #print("shape: ", logits[i].shape, inputs["labels"][:, i].shape)
                loss += self.rank_loss(logits[i].squeeze(1), inputs["labels"][:, i].clone()) / smtid_length
        else:
            loss = self.rank_loss(logits.view(bz*smtid_length, -1), inputs["labels"].view(-1))

        return {
            "rank": loss 
        }

if __name__ == "__main__":
    pass 