import sys 
import os 
import pickle 
import argparse
import glob
import logging
import time
import random
import yaml
from pathlib import Path
import shutil
from collections import OrderedDict
import warnings

import faiss
import torch
import numpy as np 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, HfArgumentParser
import torch.distributed as dist
import ujson 
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model
from ..losses.regulariaztion import L0, L1

from ..utils.utils import is_first_worker, makedir

class HNSWIndexer:
    def __init__(self, model, args):
        self.index_dir = args.index_dir
        if is_first_worker():
            if self.index_dir is not None:
                makedir(self.index_dir)
        self.model = model
        self.model.eval()
        self.args = args
    def store_embs():
        raise NotImplementedError
    
    @staticmethod
    def index(mmap_dir, index_dir, num_links=100, ef_construct=128):
        # 16, 100 --> 0.248  26.36 GB
        # 64, 128 --> 0.298, 29.65 GB
        # 100, 128 --> 0.318 (Recall@10 0.570), 32.02GB 
        doc_embeds = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32,mode="r").reshape(-1,768)

        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False 

        faiss.omp_set_num_threads(32)

        print("num_links: ", num_links, "ef_construct: ", ef_construct)
        index = faiss.IndexHNSWFlat(768, num_links, faiss.METRIC_INNER_PRODUCT)  # 768 is the dimension of vectors, 32 is a typical value for the HNSW M parameter
        index.hnsw.efConstruction = ef_construct # Typically efConstruction is set > M. A reasonable value could be 2*M
        index.verbose=True

        res = faiss.StandardGpuResources()  # Set up resources for GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Move the index to GPU. Here, "0" is the GPU ID.

        index.add(doc_embeds)

        faiss.write_index(index, os.path.join(index_dir, "model.index"))

    @staticmethod
    def index_from_flat_index(flat_index_dir, index_dir, num_links=64, ef_construct=128):
        # 16, 100 --> 0.248  26.36 GB
        # 64, 128 --> 0.298, 29.65 GB
        # 100, 128 --> 0.318 (Recall@10 0.570), 32.02GB 
        bf_index = faiss.read_index(os.path.join(flat_index_dir, 'index'))
        vectors = bf_index.reconstruct_n(0, bf_index.ntotal)

        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False 

        faiss.omp_set_num_threads(32)

        print("num_links: ", num_links, "ef_construct: ", ef_construct)
        vectors = np.array(vectors, dtype='float32')
        print(vectors.shape)
        index = faiss.IndexHNSWFlat(768, num_links, faiss.METRIC_INNER_PRODUCT)  # 768 is the dimension of vectors, 32 is a typical value for the HNSW M parameter
        index.hnsw.efConstruction = ef_construct # Typically efConstruction is set > M. A reasonable value could be 2*M
        index.verbose = True
        res = faiss.StandardGpuResources()  # Set up resources for GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)  # Move the index to GPU. Here, "0" is the GPU ID.

        index.add(vectors)

        faiss.write_index(index, os.path.join(index_dir, "model.index"))
        shutil.copy(os.path.join(flat_index_dir, 'docid'), os.path.join(index_dir, 'docid'))
    
    
    def search(self, collection_dataloader, topk, index_path, out_dir, index_ids_path):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_dataloader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs

        index = faiss.read_index(index_path)
        index = self._convert_index_to_gpu(index, list(range(8)), False)
        idx_to_docid = self._read_text_ids(index_ids_path)
        # hnsw might reuturn -1 
        idx_to_docid[-1] = "0"
        
        qid_to_rankdata = {}
        all_scores, all_idxes = index.search(query_embs, topk)
        
        for qid, scores, idxes in zip(query_ids, all_scores, all_idxes):
            docids = [idx_to_docid[idx] for idx in idxes]
            qid_to_rankdata[str(qid)] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[str(qid)][str(docid)] = float(score)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    
    def _read_text_ids(self, text_ids_path):
        idx_to_docid = {}
        with open(text_ids_path) as fin:
            for idx, line in enumerate(fin):
                docid = line.strip()
                idx_to_docid[idx] = docid
        
        print("size of idx_to_docid = {}".format(len(idx_to_docid)))
        return idx_to_docid
    
    def _get_embeddings_from_scratch(self, collection_loader, use_fp16=False, is_query=True):
        model = self.model
        
        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    #reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)


        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids
    
    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index

class ProductQuantizeIndexer:
    def __init__(self, model, args):
        self.index_dir = args.index_dir
        if is_first_worker():
            if self.index_dir is not None:
                makedir(self.index_dir)
        self.model = model
        self.model.eval()
        self.args = args

    def store_embs(self, collection_loader, local_rank, chunk_size=50_000, use_fp16=False, is_query=False):
        model = self.model
        index_dir = self.index_dir
        write_freq = chunk_size // collection_loader.batch_size
        if is_first_worker():
            print("write_freq: {}, batch_size: {}, chunk_size: {}".format(write_freq, collection_loader.batch_size, chunk_size))
        
        embeddings = []
        embeddings_ids = []
        
        chunk_idx = 0
        for idx, batch in tqdm(enumerate(collection_loader), disable= not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                                total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(model.device) for k, v in batch.items() if k != "id"}
                    if is_query:
                        raise NotImplementedError
                    else:
                        reps = unwrap_model(model).doc_encode(**inputs)
                    #reps = model(**inputs)            
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)

            if (idx + 1) % write_freq == 0:
                embeddings = np.concatenate(embeddings)
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
                assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

                text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
                id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
                np.save(text_path, embeddings)
                np.save(id_path, embeddings_ids)

                del embeddings, embeddings_ids
                embeddings, embeddings_ids = [], []

                chunk_idx += 1 
        
        if len(embeddings) != 0:
            embeddings = np.concatenate(embeddings)
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))
            print("last embedddings shape = {}".format(embeddings.shape))
            text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
            id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            chunk_idx += 1 
            
        plan = {"nranks": dist.get_world_size(), "num_chunks": chunk_idx, "index_path": os.path.join(index_dir, "model.index")}
        print("plan: ", plan)
        
        if is_first_worker():
            with open(os.path.join(self.index_dir, "plan.json"), "w") as fout:
                ujson.dump(plan, fout)

    @staticmethod
    def index(mmap_dir, num_subvectors_for_pq, index_dir, nbits=8):
        doc_embeds = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32,mode="r").reshape(-1,768)
        M = num_subvectors_for_pq
        nbits = nbits
        print("start product quantization with M = {}, nbits = {}".format(M, nbits))

        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False 

        faiss.omp_set_num_threads(32)
        #index = faiss.index_factory(doc_embeds.shape[1], f"OPQ{M},PQ{M}x8", faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexPQ(doc_embeds.shape[1], M, nbits, faiss.METRIC_INNER_PRODUCT)

        print("start training")
        index.verbose = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)    
        index.train(doc_embeds)
        index.add(doc_embeds)
        index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, os.path.join(index_dir, "model.index"))
    
    def search(self, collection_dataloader, topk, index_path, out_dir, index_ids_path):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_dataloader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs

        index = faiss.read_index(index_path)
        index = self._convert_index_to_gpu(index, list(range(8)), False)
        idx_to_docid = self._read_text_ids(index_ids_path)
        
        qid_to_rankdata = {}
        all_scores, all_idxes = index.search(query_embs, topk)
        for qid, scores, idxes in zip(query_ids, all_scores, all_idxes):
            docids = [idx_to_docid[idx] for idx in idxes]
            qid_to_rankdata[str(qid)] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[str(qid)][str(docid)] = float(score)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    
    def _read_text_ids(self, text_ids_path):
        idx_to_docid = {}
        with open(text_ids_path) as fin:
            for idx, line in enumerate(fin):
                docid = line.strip()
                idx_to_docid[idx] = docid
        
        print("size of idx_to_docid = {}".format(len(idx_to_docid)))
        return idx_to_docid

    def _get_embeddings_from_scratch(self, collection_loader, use_fp16=False, is_query=True):
        model = self.model
        
        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    #reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)


        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids

    def _index_retrieve(self, index, query_embeddings, topk, batch=None):
        if batch is None:
            nn_scores, nearest_neighbors = index.search(query_embeddings, topk)
        else:
            query_offset_base = 0
            pbar = tqdm(total=len(query_embeddings))
            nearest_neighbors = []
            nn_scores = []
            while query_offset_base < len(query_embeddings):
                batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
                batch_nn_scores, batch_nn = index.search(batch_query_embeddings, topk)
                nearest_neighbors.extend(batch_nn.tolist())
                nn_scores.extend(batch_nn_scores.tolist())
                query_offset_base += len(batch_query_embeddings)
                pbar.update(len(batch_query_embeddings))
            pbar.close()

        return nn_scores, nearest_neighbors

    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index

class AddictvieQuantizeIndexer(ProductQuantizeIndexer):
    def __init__(self, model, args):
        super().__init__(model, args)

    @staticmethod
    def index(mmap_dir, num_subvectors_for_pq, index_dir, codebook_bits):
        doc_embeds = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32,mode="r").reshape(-1,768)
        M = num_subvectors_for_pq
        codebook_bits = codebook_bits
        print("M: {}, codebook_bits: {}".format(M, codebook_bits))
        faiss.omp_set_num_threads(32)
        index = faiss.IndexResidualQuantizer(doc_embeds.shape[1], M, codebook_bits, faiss.METRIC_INNER_PRODUCT)

        print("start training")
        print("shape of doc_embeds: ", doc_embeds.shape)
        index.verbose = True
        index.train(doc_embeds)
        index.add(doc_embeds)

        faiss.write_index(index, os.path.join(index_dir, "model.index"))

    def search(self, collection_dataloader, topk, index_path, out_dir, index_ids_path):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_dataloader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs

        index = faiss.read_index(index_path)
        #index = self._convert_index_to_gpu(index, list(range(8)), False)
        idx_to_docid = self._read_text_ids(index_ids_path)
        
        qid_to_rankdata = {}
        all_scores, all_idxes = index.search(query_embs, topk)
        for qid, scores, idxes in zip(query_ids, all_scores, all_idxes):
            docids = [idx_to_docid[idx] for idx in idxes]
            qid_to_rankdata[str(qid)] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[str(qid)][str(docid)] = float(score)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

    def flat_index_search(self, collection_loader, topk, index_path, out_dir):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_loader, use_fp16=False, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs
        
        index = faiss.read_index(index_path)
        index = self._convert_index_to_gpu(index, list(range(8)), False)
        
        nn_scores, nn_doc_ids = index.search(query_embs, topk)

        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_ranks, fout)

class DenseIndexing:
    def __init__(self, model, args):
        self.index_dir = args.index_dir
        if is_first_worker():
            if self.index_dir is not None:
                makedir(self.index_dir)
        self.model = model
        self.model.eval()
        self.args = args

    def index(self, collection_loader, use_fp16=False, hidden_dim=768):
        model = self.model
        
        index = faiss.IndexFlatIP(hidden_dim)
        index = faiss.IndexIDMap(index)
        
        for idx, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    reps = model(**inputs)          
                    text_ids = batch["id"].numpy()
            
            index.add_with_ids(reps.cpu().numpy().astype(np.float32), text_ids)
            
        return index
                    
    def store_embs(self, collection_loader, local_rank, chunk_size=50_000, use_fp16=False, is_query=False):
        model = self.model
        index_dir = self.index_dir
        write_freq = chunk_size // collection_loader.batch_size
        if is_first_worker():
            print("write_freq: {}, batch_size: {}, chunk_size: {}".format(write_freq, collection_loader.batch_size, chunk_size))
        
        embeddings = []
        embeddings_ids = []
        
        chunk_idx = 0
        for idx, batch in tqdm(enumerate(collection_loader), disable= not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                                total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(model.device) for k, v in batch.items() if k != "id"}
                    if is_query:
                        raise NotImplementedError
                    else:
                        reps = unwrap_model(model).doc_encode(**inputs)
                    #reps = model(**inputs)            
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)

            if (idx + 1) % write_freq == 0:
                embeddings = np.concatenate(embeddings)
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
                assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

                text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
                id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
                np.save(text_path, embeddings)
                np.save(id_path, embeddings_ids)

                del embeddings, embeddings_ids
                embeddings, embeddings_ids = [], []

                chunk_idx += 1 
        
        if len(embeddings) != 0:
            embeddings = np.concatenate(embeddings)
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))
            print("last embedddings shape = {}".format(embeddings.shape))
            text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
            id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            chunk_idx += 1 
            
        plan = {"nranks": dist.get_world_size(), "num_chunks": chunk_idx, "index_path": os.path.join(index_dir, "model.index")}
        print("plan: ", plan)
        
        if is_first_worker():
            with open(os.path.join(self.index_dir, "plan.json"), "w") as fout:
                ujson.dump(plan, fout)

    def stat_sparse_project_encoder(self, collection_loader, use_fp16=False, apply_log_relu_logit=False):
        model = self.model 
        l0_scores, l1_scores = [], []
        docid_to_info = {}
        l0_fn = L0()
        l1_fn = L1()
        for i, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.to(model.device) for k, v in batch.items() if k != "id"}
                    _, logits = unwrap_model(model).doc_encode_and_logit(**inputs)

            if apply_log_relu_logit:
                l0_scores.append(l0_fn(
                    torch.log(1 + torch.relu(logits))
                    ).cpu().item())
                l1_scores.append(l1_fn(
                    torch.log(1 + torch.relu(logits))
                    ).cpu().item())
            else:
                l0_scores.append(l0_fn(logits).cpu().item())
                l1_scores.append(l1_fn(logits).cpu().item())

            top_scores, top_cids = torch.topk(logits, k=128, dim=1) 
            top_scores, top_cids = top_scores.cpu().tolist(), top_cids.cpu().tolist()
            for docid, scores, cids in zip(batch["id"], top_scores, top_cids):
                docid_to_info[docid.cpu().item()] = {"scores": scores, "cids": cids}
        
        return docid_to_info, np.mean(l0_scores), np.mean(l1_scores)
            

    @staticmethod
    def aggregate_embs_to_index(index_dir):
        with open(os.path.join(index_dir, "plan.json")) as fin:
            plan = ujson.load(fin)
            
        print("index_dir is: {}".format(index_dir))
        print("plan: ", plan)
        
        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]
        
        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(np.load(os.path.join(index_dir,"embs_{}_{}.npy".format(i, chunk_idx))))
                text_ids.append(np.load(os.path.join(index_dir,"ids_{}_{}.npy".format(i, chunk_idx))))

        text_embs = np.concatenate(text_embs)
        text_embs = text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)
        
        index = faiss.IndexFlatIP(text_embs.shape[1])
        index = faiss.IndexIDMap(index)

        #assert isinstance(text_ids, list)
        #text_ids = np.array(text_ids)

        index.add_with_ids(text_embs, text_ids)
        faiss.write_index(index, index_path)

        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(index_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(index_dir,"embs_{}_{}.npy".format(i, chunk_idx)))
                os.remove(os.path.join(index_dir,"ids_{}_{}.npy".format(i, chunk_idx)))
                
    @staticmethod
    def aggregate_embs_to_mmap(mmap_dir):
        with open(os.path.join(mmap_dir, "plan.json")) as fin:
            plan = ujson.load(fin)
            
        print("mmap_dir is: {}".format(mmap_dir))
        print("plan: ", plan)
        
        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]
        
        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(np.load(os.path.join(mmap_dir,"embs_{}_{}.npy".format(i, chunk_idx))))
                text_ids.append(np.load(os.path.join(mmap_dir,"ids_{}_{}.npy".format(i, chunk_idx))))

        text_embs = np.concatenate(text_embs)
        text_embs = text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)
        
        fp = np.memmap(os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32, mode='w+', shape=text_embs.shape)

        total_num = 0
        chunksize = 5_000
        for i in range(0, len(text_embs), chunksize):
            # generate some data or load a chunk of your data here. Replace `np.random.rand(chunksize, shape[1])` with your data.
            data_chunk = text_embs[i: i+chunksize]
            total_num += len(data_chunk)
            # make sure that the last chunk, which might be smaller than chunksize, is handled correctly
            if data_chunk.shape[0] != chunksize:
                fp[i:i+data_chunk.shape[0]] = data_chunk
            else:
                fp[i:i+chunksize] = data_chunk
        assert total_num == len(text_embs), (total_num, len(text_embs))
        
        with open(os.path.join(mmap_dir, "text_ids.tsv"), "w") as fout:
            for tid in text_ids:
                fout.write(f"{tid}\n")
        
        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(mmap_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(mmap_dir,"embs_{}_{}.npy".format(i, chunk_idx)))
                os.remove(os.path.join(mmap_dir,"ids_{}_{}.npy".format(i, chunk_idx)))        
                
class DenseRetriever:
    def __init__(self, model, args, dataset_name, is_beir=False):
        self.index_dir = args.index_dir
        self.out_dir = os.path.join(args.out_dir, dataset_name) if (dataset_name is not None and not is_beir) \
            else args.out_dir
        if is_first_worker():
            makedir(self.out_dir)
        if self.index_dir is not None:
            self.index_path = os.path.join(self.index_dir, "model.index")
        self.model = model
        self.model.eval()
        self.args = args
        
    def retrieve(self, collection_loader, topk, save_run=True, index=None, use_fp16=False):
        query_embs, query_ids = self._get_embeddings_from_scratch(collection_loader, use_fp16=use_fp16, is_query=True)
        query_embs = query_embs.astype(np.float32) if query_embs.dtype==np.float16 else query_embs
        

        if index is None:
            index = faiss.read_index(self.index_path)
            index = self._convert_index_to_gpu(index, list(range(8)), False)
        
        nn_scores, nn_doc_ids = self._index_retrieve(index, query_embs, topk, batch=128)
        
        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)
        
        if save_run:
            with open(os.path.join(self.out_dir, "run.json"), "w") as fout:
                ujson.dump(qid_to_ranks, fout)
            return {"retrieval": qid_to_ranks}
        else:
            return {"retrieval": qid_to_ranks}

    @staticmethod
    def get_first_smtid(model, collection_loader, use_fp16=False, is_query=True):
        model = model.base_model 
        print("the flag for model.decoding: ", model.config.decoding)

        text_ids = []
        semantic_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    logits = model(**inputs).logits[0] #[bz, vocab]
            smtids = torch.argmax(logits, dim=1).cpu().tolist() 
            text_ids.extend(batch["id"].tolist())
            semantic_ids.extend(smtids)
        
        return text_ids, semantic_ids
        
    def _get_embeddings_from_scratch(self, collection_loader, use_fp16=False, is_query=True):
        model = self.model
        
        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
                    #reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)


        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids
    
    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024*1024*1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index
    
    def _index_retrieve(self, index, query_embeddings, topk, batch=None):
        if batch is None:
            nn_scores, nearest_neighbors = index.search(query_embeddings, topk)
        else:
            query_offset_base = 0
            pbar = tqdm(total=len(query_embeddings))
            nearest_neighbors = []
            nn_scores = []
            while query_offset_base < len(query_embeddings):
                batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
                batch_nn_scores, batch_nn = index.search(batch_query_embeddings, topk)
                nearest_neighbors.extend(batch_nn.tolist())
                nn_scores.extend(batch_nn_scores.tolist())
                query_offset_base += len(batch_query_embeddings)
                pbar.update(len(batch_query_embeddings))
            pbar.close()

        return nn_scores, nearest_neighbors

    
class GenDocIDRetriever:
    def __init__(self, model, args, dataset_name, is_beir=False):
        self.index_dir = args.index_dir
        self.out_dir = os.path.join(args.out_dir, dataset_name) if (dataset_name is not None and not is_beir) \
            else args.out_dir
        if is_first_worker():
            makedir(self.out_dir)
        if self.index_dir is not None:
            self.index_path = os.path.join(self.index_dir, "model.index")
        self.model = model
        self.model.eval()
        self.args = args 
     
class DenseApproxEval:
    def __init__(self, model, args, collection_loader, q_loader, **kwargs):
        self.model = model
        self.model.eval()
        if is_first_worker():
            makedir(args.out_dir)
            
        self.args = args
        self.collection_loader = collection_loader 
        self.q_loader = q_loader
        
    def index_and_retrieve(self, i):
        # i is useless
        indexer = DenseIndexing(self.model, self.args)
        index_d = indexer.index(self.collection_loader)
        retriver = DenseRetriever(self.model, self.args, dataset_name=None)
        
        return retriver.retrieve(self.q_loader, self.args.top_k, save_run=True, index=index_d)


    
