import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from .t5_generative_retriever import T5forDocIDConfig, T5ForDocIDGeneration

class CrossEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

        self.cls_loss = nn.BCEWithLogitsLoss()
        self.model_args = None # incompatible with previous models
        
    def forward(self, **inputs):
        logits = self.model(**inputs["qd_kwargs"]).logits.view(-1)
        if "labels" in inputs:            
            loss = self.cls_loss(logits, inputs["labels"])
            return {"cls": loss}
        else:
            return logits
    
    def rerank_forward(self, qd_kwargs):
        out = {}
        scores = self.model(**qd_kwargs).logits.view(-1)
        out.update({"scores": scores})
        
        return out
    
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        return cls(model_name_or_path)
    
    def save_pretrained(self, save_dir):
        self.model.save_pretrained(save_dir)

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.out_proj = nn.Linear(config.d_model, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5SeqCrossEncoder(nn.Module):
    def __init__(self, model_name_or_path, shared_output_input_embeds):
        super().__init__() 
        config = T5forDocIDConfig().from_pretrained(model_name_or_path)
        config.decoding = False

        if shared_output_input_embeds is not None:
            assert shared_output_input_embeds in [False, True]
            config.shared_output_input_embeds = shared_output_input_embeds

        self.base_model = T5ForDocIDGeneration.from_pretrained(model_name_or_path, config=config)
        self.config = config
        self.model_args = None # incompatible with previous models

        self.head = T5ClassificationHead(config)

        self.cls_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, seq_length] + [bz, smtid_length]
            doc_encoding: [bz, smtid_length]
            labels: [bz]
        Returns:
            logits: [bz]
        """
        last_rep = self.base_model(**inputs["tokenized_query"]).decoder_last_hidden_state #[bz, smtid_length, d_model]

        logits = self.head(last_rep.mean(dim=1)).view(-1)

        if "labels" in inputs:            
            loss = self.cls_loss(logits, inputs["labels"])
            return {"cls": loss}
        else:
            return logits
    
    def rerank_forward(self, **inputs):
        out = {}
        scores = self.base_model(**inputs).logits.view(-1)
        out.update({"scores": scores})
        
        return out
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, shared_output_input_embeds=None):
        return cls(model_name_or_path, shared_output_input_embeds)
    
    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)
        