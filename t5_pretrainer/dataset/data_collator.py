from typing import Any
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy

from ..utils.utils import flatten_list

# for training   
class LngKnpMarginMSEforT5SeqAQCollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        if len(batch[0]) == 10:
            q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg = [list(x) for x in zip(*batch)]
        elif len(batch[0]) == 12:
            q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg, \
            smtid_8_s_pos, smtid_8_s_neg = [list(x) for x in zip(*batch)]
        elif len(batch[0]) == 14:
            q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg, \
            smtid_8_s_pos, smtid_8_s_neg, smtid_16_s_pos, smtid_16_s_neg = [list(x) for x in zip(*batch)]
        else:
            raise ValueError("element in batch don't have deisred len since its length is {}".format(len(batch[0])))

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        q_pos["decoder_input_ids"] = torch.LongTensor(q_pos_decoder_input_ids)
        q_neg["decoder_input_ids"] = torch.LongTensor(q_neg_decoder_input_ids)

        if len(batch[0]) == 10:
            return {
                "pos_tokenized_query": q_pos,
                "neg_tokenized_query": q_neg,
                "pos_doc_encoding": torch.LongTensor(pos_doc_encoding),
                "neg_doc_encoding": torch.LongTensor(neg_doc_encoding),
                "teacher_pos_scores": torch.FloatTensor(s_pos),
                "teacher_neg_scores": torch.FloatTensor(s_neg),
                "smtid_4_teacher_pos_scores": torch.FloatTensor(smtid_4_s_pos),
                "smtid_4_teacher_neg_scores": torch.FloatTensor(smtid_4_s_neg),
            }
        elif len(batch[0]) == 12:
            return {
                "pos_tokenized_query": q_pos,
                "neg_tokenized_query": q_neg,
                "pos_doc_encoding": torch.LongTensor(pos_doc_encoding),
                "neg_doc_encoding": torch.LongTensor(neg_doc_encoding),
                "teacher_pos_scores": torch.FloatTensor(s_pos),
                "teacher_neg_scores": torch.FloatTensor(s_neg),
                "smtid_4_teacher_pos_scores": torch.FloatTensor(smtid_4_s_pos),
                "smtid_4_teacher_neg_scores": torch.FloatTensor(smtid_4_s_neg),
                "smtid_8_teacher_pos_scores": torch.FloatTensor(smtid_8_s_pos),
                "smtid_8_teacher_neg_scores": torch.FloatTensor(smtid_8_s_neg),
            }
        else:
            return {
                "pos_tokenized_query": q_pos,
                "neg_tokenized_query": q_neg,
                "pos_doc_encoding": torch.LongTensor(pos_doc_encoding),
                "neg_doc_encoding": torch.LongTensor(neg_doc_encoding),
                "teacher_pos_scores": torch.FloatTensor(s_pos),
                "teacher_neg_scores": torch.FloatTensor(s_neg),
                "smtid_4_teacher_pos_scores": torch.FloatTensor(smtid_4_s_pos),
                "smtid_4_teacher_neg_scores": torch.FloatTensor(smtid_4_s_neg),
                "smtid_8_teacher_pos_scores": torch.FloatTensor(smtid_8_s_pos),
                "smtid_8_teacher_neg_scores": torch.FloatTensor(smtid_8_s_neg),
                "smtid_16_teacher_pos_scores": torch.FloatTensor(smtid_16_s_pos),
                "smtid_16_teacher_neg_scores": torch.FloatTensor(smtid_16_s_neg)
            }

class Seq2SeqForT5SeqAQCollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        print("self_max_length", self.max_length)
    
    def __call__(self, batch):
        query, query_decoder_input_ids, query_smtids = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(query,
                                        add_special_tokens=True,
                                        padding="longest",  # pad to max sequence length in batch
                                        truncation="longest_first",  # truncates to self.max_length
                                        max_length=self.max_length,
                                        return_attention_mask=True,
                                        return_tensors="pt") #[bz, seq_length]
        
        query_smtids = torch.LongTensor(query_smtids) # [bz, smtid_len]
        tokenized_query["decoder_input_ids"] = torch.LongTensor(query_decoder_input_ids)

        return {
            "tokenized_query": tokenized_query,
            "labels": query_smtids,
        }

class MarginMSEforT5SeqAQCollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
        q_pos_decoder_input_ids, q_neg_decoder_input_ids = [list(x) for x in zip(*batch)]

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        q_pos["decoder_input_ids"] = torch.LongTensor(q_pos_decoder_input_ids)
        q_neg["decoder_input_ids"] = torch.LongTensor(q_neg_decoder_input_ids)

        return {
            "pos_tokenized_query": q_pos,
            "neg_tokenized_query": q_neg,
            "pos_doc_encoding": torch.LongTensor(pos_doc_encoding),
            "neg_doc_encoding": torch.LongTensor(neg_doc_encoding),
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg)
        }
 
class MarginMSEforPretrainCollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        if len(batch[0]) == 10:
            q_pos, q_neg, d_pos, d_neg, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids = [list(x) for x in zip(*batch)]
        elif len(batch[0]) == 12:
            q_pos, q_neg, d_pos, d_neg, pos_prev_smtids, neg_prev_smtids, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids = [list(x) for x in zip(*batch)]
        else:
            raise ValueError(f"the batch example with size {len(batch[0])} is not valid.")

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        d_pos = self.tokenizer(d_pos,
                              add_special_tokens=True,
                            padding="longest",  # pad to max sequence length in batch
                            truncation="longest_first",  # truncates to self.max_length
                            max_length=self.max_length,
                            return_attention_mask=True,
                            return_tensors="pt")
        d_neg = self.tokenizer(d_neg,
                               add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        
        q_pos["decoder_input_ids"] = torch.LongTensor(d_pos_decoder_input_ids)
        q_neg["decoder_input_ids"] = torch.LongTensor(d_neg_decoder_input_ids)
        d_pos["decoder_input_ids"] = torch.LongTensor(q_pos_decoder_input_ids)
        d_neg["decoder_input_ids"] = torch.LongTensor(q_neg_decoder_input_ids)

        if len(batch[0]) == 10:
            return {
            "tokenized_pos_query": q_pos,
            "tokenized_neg_query": q_neg,
            "tokenized_pos_doc": d_pos,
            "tokenized_neg_doc": d_neg,

            "teacher_pos_score": torch.FloatTensor(s_pos),
            "teacher_neg_score": torch.FloatTensor(s_neg),
            }
        else:
            return {
            "tokenized_pos_query": q_pos,
            "tokenized_neg_query": q_neg,
            "tokenized_pos_doc": d_pos,
            "tokenized_neg_doc": d_neg, 

            "pos_prev_smtids": torch.LongTensor(pos_prev_smtids),
            "neg_prev_smtids": torch.LongTensor(neg_prev_smtids),

            "teacher_pos_score": torch.FloatTensor(s_pos),
            "teacher_neg_score": torch.FloatTensor(s_neg),
        }
