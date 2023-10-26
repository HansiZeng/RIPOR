import os

import torch
import torch.nn as nn
from tqdm import tqdm
import ujson
import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

from ..utils.utils import is_first_worker, makedir

class Reranker:
    def __init__(self, model, dataloader, config, dtype=None, dataset_name=None, is_beir=False, write_to_disk=False, local_rank=-1):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.model.eval()
        
        self.write_to_disk = write_to_disk
        if write_to_disk:
            self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
            if is_first_worker():
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
        
        self.dtype = dtype
        self.local_rank = local_rank
        

    def reranking(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["pair_ids"], list)
                    all_pair_ids.extend(batch["pair_ids"])
                        
        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def reranking_for_same_prefix_pair(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True, prefix_name=None):
        all_triple_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["triple_ids"], list)
                    all_triple_ids.extend(batch["triple_ids"])
                        
        qid_to_rankdata = self.triple_ids_to_json_output(all_scores, all_triple_ids)
        
        if self.write_to_disk:
            if prefix_name is not None:
                out_path = os.path.join(self.out_dir, "{}{}.json".format(prefix_name, f"_{name}" if name is not None else ""))
            else:
                out_path = os.path.join(self.out_dir, "qid_to_smtid_to_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write qid_to_smtid_to_rerank to path: {}".format(out_path))
            #print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
            #                                                  np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def query_to_smtid_reranking(self, name=None, use_fp16=False, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {"tokenized_query": self.kwargs_to_cuda(batch["tokenized_query"]),
                     "labels": batch["labels"].to(self.local_rank)}
                    if hasattr(self.model, "module"):
                        scores = self.model.module.get_query_smtids_score(**inputs) #[bz, seq_len]
                    else:
                        scores = self.model.get_query_smtids_score(**inputs) #[bz, seq_len]

            scores = torch.sum(scores, dim=-1).cpu().tolist() #[bz]
            all_scores.extend(scores)
            all_pair_ids.extend(batch["pair_ids"])

        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "qid_smtids_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata
    
    def cond_prev_smtid_t5seq_encoder_reranking(self, name=None, use_fp16=False, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {
                        "tokenized_query": self.kwargs_to_cuda(batch["tokenized_query"]),
                        "tokenized_doc": self.kwargs_to_cuda(batch["tokenized_doc"])}
                    inputs["prev_smtids"] = batch["prev_smtids"].to(self.local_rank)
                    if hasattr(self.model, "module"):
                        scores = self.model.module.cond_prev_smtid_query_doc_score(**inputs) #[bz, seq_len]
                    else:
                        scores = self.model.cond_prev_smtid_query_doc_score(**inputs) #[bz, seq_len]

            all_scores.extend(scores.cpu().tolist())
            all_pair_ids.extend(batch["pair_ids"])

        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "cond_prev_smtid_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def assign_scores_for_pseudo_queries(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), disable=self.local_rank>0):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["pair_ids"], list)
                    all_pair_ids.extend(batch["pair_ids"])
                        
        pid_to_rankdata = self.pair_ids_to_docid_to_qids_json_output(all_scores, all_pair_ids)
        
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "pid_qids_rerank_scores{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} docs, avg_doc_length = {:.3f}".format(len(pid_to_rankdata), 
                                                              np.mean([len(xs) for xs in pid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(pid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return pid_to_rankdata
    
    def kwargs_to_cuda(self, input_kwargs):
        assert isinstance(input_kwargs, BatchEncoding) or isinstance(input_kwargs, dict), (type(input_kwargs))
        
        if self.local_rank != -1:
            return {k:v.to(self.local_rank) for k, v in input_kwargs.items()}
        else:
            return {k:v.cuda() for k, v in input_kwargs.items()}
    
    @staticmethod
    def pair_ids_to_json_output(scores, pair_ids):
        qid_to_rankdata = {}
        assert len(scores) == len(pair_ids)
        for i in range(len(scores)):
            s = float(scores[i])
            qid, pid = pair_ids[i]
            qid, pid = str(qid), str(pid)
            
            if qid not in qid_to_rankdata:
                qid_to_rankdata[qid] = {pid: s}
            else:
                qid_to_rankdata[qid][pid] = s
                
        return qid_to_rankdata
    
    @staticmethod
    def triple_ids_to_json_output(scores, triple_ids):
        qid_to_smtid_to_rankdata = {}
        for i in range(len(scores)):
            s = float(scores[i])
            qid, docid, smtid = triple_ids[i]
            qid, docid, smtid = str(qid), str(docid), str(smtid)

            if qid not in qid_to_smtid_to_rankdata:
                qid_to_smtid_to_rankdata[qid] = {smtid: [(docid, s)]}
            else:
                if smtid not in qid_to_smtid_to_rankdata[qid]:
                    qid_to_smtid_to_rankdata[qid][smtid] = [(docid, s)]
                else:
                    qid_to_smtid_to_rankdata[qid][smtid] += [(docid, s)]
        return qid_to_smtid_to_rankdata

    @staticmethod
    def pair_ids_to_docid_to_qids_json_output(scores, pair_ids):
        pid_to_rankdata = {}
        assert len(scores) == len(pair_ids)
        for i in range(len(scores)):
            s = float(scores[i])
            qid, pid = pair_ids[i]
            qid, pid = str(qid), str(pid)

            if pid not in pid_to_rankdata:
                pid_to_rankdata[pid] = {qid: s}
            else:
                pid_to_rankdata[pid][qid] = s 

        return pid_to_rankdata