import os 

import transformers 
from transformers import T5Model, AutoTokenizer
from transformers.trainer import Trainer
import torch
import torch.nn as nn

#from ..losses import KLDivLoss

class T5ModelEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args 

        self.t5_model = T5Model.from_pretrained(model_args.model_name_or_path)
        self.t5_config = self.t5_model.config
        
        self.start_token_id, self.pad_token_id, self.eos_token_id = self.t5_config.decoder_start_token_id, self.t5_config.pad_token_id, self.t5_config.eos_token_id
        assert self.start_token_id == self.pad_token_id == 0 and self.eos_token_id == 1, (self.start_token_id, self.pad_token_id, self.eos_token_id)

        #decoder_embed_tokens = nn.Embedding(model_args.decoder_vocab_size, model_args.d_model)
        #decoder_embed_tokens.weight[self.start_token_id] = self.t5_model.shared.weight[self.start_token_id]
        #decoder_embed_tokens.weight[self.eos_token_id] = self.t5_model.shared.weight[self.eos_token_id]
        #self.t5_model.decoder.set_input_embeddings(decoder_embed_tokens.weight)


    def encode(self, input_ids, attention_mask):
        dec_input_ids = torch.full((len(input_ids), self.model_args.max_decoder_length), self.pad_token_id).long().to(input_ids.device)
        dec_input_ids[:, 0] = self.start_token_id 
        last_hidden_states = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_input_ids,
                      output_hidden_states=True).last_hidden_state
        return last_hidden_states[:, 0, :]

    def forward(self, **inputs):
        return self.encode(**inputs)
    
    @classmethod
    def from_pretrained(cls, model_path, model_args=None):
        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, "model.pt")
            model_args_path = os.path.join(model_path, "model_args.pt")
            model_args = torch.load(model_args_path)
        elif os.path.isfile(model_path):
            assert model_args != None 
            ckpt_path = model_path
        else:
            raise ValueError("model_path: {} is not expected".format(model_path))
        
        model = cls(model_args)
        print("load pretrained model from local path {}".format(model_path))

        model_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_dict)

        return model 

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        model_to_save = self
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(self.model_args, os.path.join(save_dir, "model_args.pt"))

class MarginMSET5ModelEncoder(T5ModelEncoder):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.loss = nn.MSELoss()

    def forward(self, **inputs):
        query_reps = self.encode(**inputs["enc_query"]) #[bz, d_model]
        pos_doc_reps = self.encode(**inputs["enc_pos_doc"])
        neg_doc_reps = self.encode(**inputs["enc_neg_doc"])

        student_pos_scores = torch.sum(query_reps * pos_doc_reps, dim=-1)
        student_neg_scores = torch.sum(query_reps * neg_doc_reps, dim=-1)
        margin = student_pos_scores - student_neg_scores

        teacher_pos_scores, teacher_neg_scores = inputs["teacher_pos_scores"], inputs["teacher_neg_scores"]
        teacher_margin = teacher_pos_scores - teacher_neg_scores

        return self.loss(margin.squeeze(), teacher_margin.squeeze())   


class KLDivT5ModelEncoder(T5ModelEncoder):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.loss = KLDivLoss()

    def forward(self, **inputs):
        query_reps = self.encode(**inputs["enc_query"]) #[bz, d_model]
        bz, d_model = query_reps.size()
        doc_reps = self.encode(**inputs["enc_nway_docs"]).view(bz, -1, d_model) #[bz, nway, d_model]

        logits = torch.sum(query_reps.view(bz, 1, d_model) * doc_reps, dim=-1) #[bz, nway]
        loss = self.loss({"nway_scores": logits, "labels": inputs["labels"]})

        return loss 