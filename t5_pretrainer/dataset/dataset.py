import gzip
import json
import os
import pickle
import random
import copy

import ujson
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer

QUERY_PREFIX = "query: "
DOC_PREFIX = "document: "

# for rerank 
class RerankDataset(Dataset):
    def __init__(self, run_json_path, document_dir, query_dir, json_type="jsonl"):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        if json_type == "jsonl":
            self.all_pair_ids = []
            with open(run_json_path) as fin:
                for line in fin:
                    example = ujson.loads(line)
                    qid, docids = example["qid"], example["docids"]
                    for docid in docids:
                        self.all_pair_ids.append((qid, docid))
        else:
            with open(run_json_path) as fin:
                qid_to_rankdata = ujson.load(fin)
            
            self.all_pair_ids = []
            for qid, rankdata in qid_to_rankdata.items():
                for pid, _ in rankdata.items():
                    self.all_pair_ids.append((str(qid), str(pid)))
        
    def __len__(self):
        return len(self.all_pair_ids)
    
    def __getitem__(self, idx):
        pair_id = self.all_pair_ids[idx]
        qid, pid = pair_id
        
        query = self.query_dataset[qid][1]
        doc = self.document_dataset[pid][1]
        
        return {
            "pair_id": pair_id,
            "query": query,
            "doc": doc,
        }

class PseudoQueryForScoreDataset(Dataset):
    def __init__(self, document_dir, pseudo_queries_path, docid_pseudo_qids_path):
        docid_to_doc = {}
        with open(os.path.join(document_dir, "raw.tsv")) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                docid_to_doc[docid] = doc 
        self.docid_to_doc = docid_to_doc
        
        qid_to_query = {}
        with open(pseudo_queries_path) as fin:
            for line in tqdm(fin, total=442090001):
                try:
                    qid, query = line.strip().split("\t")
                    qid_to_query[qid] = query 
                except:
                    print(line)
                    raise ValueError("wrong here")
        self.qid_to_query = qid_to_query

        self.out_pairs = [] 
        with open(docid_pseudo_qids_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                docid = example["docid"]
                qids = example["qids"]

                for qid in qids:
                    self.out_pairs.append((qid, docid))
    
    def __len__(self):
        return len(self.out_pairs)

    def __getitem__(self, idx):
        qid, docid = self.out_pairs[idx]
        query, doc = self.qid_to_query[qid], self.docid_to_doc[docid]

        return (qid, docid, query, doc)

class QueryToSmtidRerankDataset(Dataset):
    def __init__(self, qid_docids_path, queries_path, docid_to_smtids, docid_to_strsmtid):
        self.examples = []
        with open(qid_docids_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(
                    {"qid": example["qid"], "docids": example["docids"]}
                )


        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query

        self.qid_strsmtid_pairs = []
        self.qid_smtids_pairs = []
        for example in self.examples:
            qid = example["qid"]
            for docid in example["docids"]:
                self.qid_strsmtid_pairs.append((qid, docid_to_strsmtid[docid]))
                self.qid_smtids_pairs.append((qid, docid_to_smtids[docid]))

    def __len__(self):
        return len(self.qid_strsmtid_pairs)

    def __getitem__(self, idx):
        qid, strsmtid = self.qid_strsmtid_pairs[idx]
        _, smtids = self.qid_smtids_pairs[idx]
        query = QUERY_PREFIX + self.qid_to_query[qid]

        assert smtids[0] == -1, smtids 
        decoder_input_ids = smtids[:-1]
        labels = smtids[1:]

        return query, qid, strsmtid, decoder_input_ids, labels

class TeacherRerankFromQidSmtidsDataset(Dataset):
    def __init__(self, qid_smtid_rank_path, docid_to_smtids_path, queries_path, collection_path):
        with open(docid_to_smtids_path) as fin:
            docid_to_smtids = ujson.load(fin)
        smtid_to_docids = {}
        for docid, smtids in docid_to_smtids.items():
            assert smtids[0] == -1, smtids 
            smtid = "_".join([str(x) for x in smtids[1:]])
            if smtid not in smtid_to_docids:
                smtid_to_docids[smtid] = [docid]
            else:
                smtid_to_docids[smtid] += [docid]
        self.docid_to_smtids = docid_to_smtids

        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query 
        
        self.docid_to_doc = {}
        with open(collection_path) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                self.docid_to_doc[docid] = doc 

        smtid_lenghts = []
        self.all_pair_ids = []
        with open(qid_smtid_rank_path) as fin:
            rank_data = ujson.load(fin)
        for qid in rank_data:
            smtid_lenghts.append(len(rank_data[qid]))
            for smtid in rank_data[qid]:
                docids = smtid_to_docids[smtid]
                for docid in docids:
                    self.all_pair_ids.append((str(qid), str(docid)))

        print("number of qids = {}".format(len(rank_data)))
        print("average number of smtids_per_query = {:.3f}".format(np.mean(smtid_lenghts)))
        print("average docs per query = {:.3f}".format(len(self.all_pair_ids) / len(rank_data)))
        

    def __len__(self):
        return len(self.all_pair_ids)
    
    def __getitem__(self, idx):
        pair_id = self.all_pair_ids[idx]
        qid, docid = pair_id
        
        query = self.qid_to_query[qid]
        doc = self.docid_to_doc[docid]
        
        return {
            "pair_id": pair_id,
            "query": query,
            "doc": doc,
        }

class CrossEncRerankForSamePrefixPair(Dataset):
    def __init__(self, qid_to_smtid_to_docids, queries_path, collection_path):
        self.qid_to_query = {}
        with open(queries_path) as fin:
            for line in fin:
                qid, query = line.strip().split("\t")
                self.qid_to_query[qid] = query 
        
        self.docid_to_doc = {}
        with open(collection_path) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                self.docid_to_doc[docid] = doc 

        self.qid_to_smtid_to_docids = qid_to_smtid_to_docids
        self.triples = []
        
        for i, qid in enumerate(tqdm(self.qid_to_smtid_to_docids, total=len(self.qid_to_smtid_to_docids))):
            for smtid, docids in self.qid_to_smtid_to_docids[qid].items():
                for docid in docids:
                    triple = (qid, docid, smtid)
                    self.triples.append(triple)


    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        qid, docid, smtid = self.triples[idx]
        query = self.qid_to_query[qid]
        doc = self.docid_to_doc[docid]

        return {
            "triple_id": (qid, docid, smtid),
            "query": query,
            "doc": doc
        }

# for evaluate
class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]

class CollectionDatasetWithDocIDPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style, tid_to_smtid_path=None, add_prefix=False, is_query=False):
        self.data_dir = data_dir
        assert id_style in ("content_id", "row_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

        if tid_to_smtid_path is not None:
            with open(tid_to_smtid_path) as fin:
                self.tid_to_smtid = ujson.load(fin)
            #assert len(self.tid_to_smtid["2"]) == 2 and self.tid_to_smtid["2"][0] == -1, (self.tid_to_smtid["2"])
        else:
            self.tid_to_smtid = {}
            if self.id_style == "row_id":
                assert len(self.line_dict) == len(self.data_dict)
                for tid in self.line_dict.values():
                    self.tid_to_smtid[tid] = [-1]
            else:
                assert len(self.line_dict) == 0
                for tid in self.data_dict:
                    self.tid_to_smtid[tid] = [-1]

        self.add_prefix = add_prefix
        self.is_query = is_query

        #print("example tid = 0 is : ", self.tid_to_smtid["0"])

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            text = self.data_dict[idx]
        else:
            text = self.data_dict[str(idx)]
        
        text = self.data_dict[idx]
        if self.add_prefix:
            if self.is_query:
                text = QUERY_PREFIX + text 
            else:
                text = DOC_PREFIX + text
        #print(len(self.tid_to_smtid), self.line_dict[idx])
        #print("text: ", text) 
        if self.id_style == "row_id":
            return self.line_dict[idx], text, self.tid_to_smtid[self.line_dict[idx]]
        else:
            return str(idx), text, self.tid_to_smtid[str(idx)]

class CollectionAQDataset(Dataset):
    def __init__(self, collection_path, docid_to_smtid_path):
        docid_to_doc = {}
        with open(collection_path) as fin:
            for line in fin:
                docid, doc = line.strip().split("\t")
                docid_to_doc[docid] = doc 
        self.doc_pairs = []
        for docid, doc in docid_to_doc.items():
            self.doc_pairs.append((docid, doc))

        with open(docid_to_smtid_path) as fin:
            docid_to_smtid = ujson.load(fin)
        self.docid_to_smtid = docid_to_smtid

    def __len__(self):
        return len(self.doc_pairs)

    def __getitem__(self, idx):
        docid, doc = self.doc_pairs[idx]

        doc_encoding = self.docid_to_smtid[docid][1:]
        
        return docid, doc_encoding

# for training 
class TripleMarginMSEDataset(Dataset):
    def __init__(self, examples_path, document_dir, query_dir, docid_to_smtid_path):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")

        self.examples = []
        with open(examples_path) as fin:
            for line in fin:
                qid, pos_docid, neg_docid, pos_score, neg_score = line.strip().split("\t")
                pos_score = float(pos_score)
                neg_score = float(neg_score)
                self.examples.append((qid, pos_docid, neg_docid, pos_score, neg_score))

        if docid_to_smtid_path is not None:
            with open(docid_to_smtid_path) as fin: 
                self.docid_to_smtid = ujson.load(fin)
            assert self.docid_to_smtid["0"][0] == -1, self.docid_to_smtid["0"]
        else:
            self.docid_to_smtid = None 

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, positive, negative, s_pos, s_neg = self.examples[idx]

        q = self.query_dataset[query][1]
        d_pos = self.document_dataset[positive][1]
        d_neg = self.document_dataset[negative][1]

        if self.docid_to_smtid is not None:
            pos_prev_smtids = self.docid_to_smtid[positive][1:]
            neg_prev_smtids = self.docid_to_smtid[negative][1:]

            d_pos_decoder_input_ids = self.docid_to_smtid[positive]
            d_neg_decoder_input_ids = self.docid_to_smtid[negative]
            q_pos_decoder_input_ids = copy.deepcopy(d_pos_decoder_input_ids)
            q_neg_decoder_input_ids = copy.deepcopy(d_neg_decoder_input_ids)
        else:
            pos_prev_smtids = None 
            neg_prev_smtids = None
            d_pos_decoder_input_ids = [-1]
            d_neg_decoder_input_ids = [-1]
            q_pos_decoder_input_ids = [-1]
            q_neg_decoder_input_ids = [-1]

        q_pos = "query: " + q.strip()
        q_neg = "query: " + q.strip()
        d_pos = "document: " + d_pos.strip()
        d_neg = "document: " + d_neg.strip()

        if pos_prev_smtids is None and neg_prev_smtids is None:
            return q_pos, q_neg, d_pos, d_neg, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids 
        else:
            return q_pos, q_neg, d_pos, d_neg, pos_prev_smtids, neg_prev_smtids, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids
        
class LngKnpMarginMSEforT5SeqAQDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 docid_to_smtid_path, smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert docid_to_smtid_path == None
        else:
            if docid_to_smtid_path is not None:
                with open(docid_to_smtid_path) as fin: 
                    self.docid_to_smtid = ujson.load(fin)
                tmp_docids = list(self.docid_to_smtid.keys())
                assert self.docid_to_smtid[tmp_docids[0]][0] == -1, self.docid_to_smtid[tmp_docids[0]]
            else:
                self.docid_to_smtid = None 

        # sanity check 
        smtid_len = len(self.examples[0]["smtids"][0].split("_"))
        if smtid_len == 32:
            print("smtid_len is 32")
            assert "smtid_4_scores" in self.examples[0] and "smtid_8_scores" in self.examples[0] and "smtid_16_scores" in self.examples[0]
        elif smtid_len == 16:
            print("smtid_len is 16")
            assert "smtid_4_scores" in self.examples[0] and "smtid_8_scores" in self.examples[0]
            assert "smtid_16_scores" not in self.examples[0] 
        elif smtid_len == 8:
            print("smtid_len is 8")
            assert "smtid_4_scores" in self.examples[0]
            assert "smtid_8_scores" not in self.examples[0] and "smtid_16_scores" not in self.examples[0] 
        else:
            raise ValueError("not valid smtid_len = {}".format(smtid_len))
        self.smtid_len = smtid_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            positive = example["smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]
        if self.smtid_len == 8:
            smtid_4_s_pos = example["smtid_4_scores"][0]
        if self.smtid_len == 16:
            smtid_4_s_pos = example["smtid_4_scores"][0]
            smtid_8_s_pos = example["smtid_8_scores"][0]
        if self.smtid_len == 32:
            smtid_4_s_pos = example["smtid_4_scores"][0]
            smtid_8_s_pos = example["smtid_8_scores"][0]
            smtid_16_s_pos = example["smtid_16_scores"][0]

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            negative = example["smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]
        if self.smtid_len == 8:
            smtid_4_s_neg = example["smtid_4_scores"][neg_idx]
        if self.smtid_len == 16:
            smtid_4_s_neg = example["smtid_4_scores"][neg_idx]
            smtid_8_s_neg = example["smtid_8_scores"][neg_idx]
        if self.smtid_len == 32:
            smtid_4_s_neg = example["smtid_4_scores"][neg_idx]
            smtid_8_s_neg = example["smtid_8_scores"][neg_idx]
            smtid_16_s_neg = example["smtid_16_scores"][neg_idx]

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            pos_smtids = [-1] + [int(x) for x in positive.split("_")]
            neg_smtids = [-1] + [int(x) for x in negative.split("_")]
            pos_doc_encoding = pos_smtids[1:]
            neg_doc_encoding = neg_smtids[1:]
            q_pos_decoder_input_ids = pos_smtids[:-1]
            q_neg_decoder_input_ids = neg_smtids[:-1]
        else:
            pos_doc_encoding = self.docid_to_smtid[str(positive)][1:]
            neg_doc_encoding = self.docid_to_smtid[str(negative)][1:]
            q_pos_decoder_input_ids = self.docid_to_smtid[str(positive)][:-1]
            q_neg_decoder_input_ids = self.docid_to_smtid[str(negative)][:-1]

        q_pos = "query: " + q.strip()
        q_neg = "query: " + q.strip()
        
        if self.smtid_len == 8:
            return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg 
        elif self.smtid_len == 16:
            return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg, smtid_8_s_pos, smtid_8_s_neg
        else:      
            return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
            q_pos_decoder_input_ids, q_neg_decoder_input_ids, smtid_4_s_pos, smtid_4_s_neg, \
            smtid_8_s_pos, smtid_8_s_neg, smtid_16_s_pos, smtid_16_s_neg

class Seq2SeqForT5SeqAQDataset(Dataset):
    def __init__(self, example_path, docid_to_smtid_path):
        with open(docid_to_smtid_path) as fin:
            docid_to_smtid = ujson.load(fin)

        self.examples = []
        with open(example_path) as fin:
            for i, line in tqdm(enumerate(fin)):
                example = ujson.loads(line)
                docid, query = example["docid"], example["query"]
                smtid = docid_to_smtid[docid]
                self.examples.append((query, smtid))
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        query, smtid = self.examples[idx]
        query_decoder_input_ids = smtid[:-1]
        query_smtids = smtid[1:]

        assert len(smtid) in [2, 5, 9, 17, 33], len(smtid)
        assert len(query_smtids) == len(query_decoder_input_ids) and query_decoder_input_ids[0] == -1
        
        return query, query_decoder_input_ids, query_smtids

class MarginMSEforT5SeqAQDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 docid_to_smtid_path, smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert docid_to_smtid_path == None
        else:
            if docid_to_smtid_path is not None:
                with open(docid_to_smtid_path) as fin: 
                    self.docid_to_smtid = ujson.load(fin)
                tmp_docids = list(self.docid_to_smtid.keys())
                assert self.docid_to_smtid[tmp_docids[0]][0] == -1, self.docid_to_smtid[tmp_docids[0]]
            else:
                self.docid_to_smtid = None 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            positive = example["smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            negative = example["smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            pos_smtids = [-1] + [int(x) for x in positive.split("_")]
            neg_smtids = [-1] + [int(x) for x in negative.split("_")]
            pos_doc_encoding = pos_smtids[1:]
            neg_doc_encoding = neg_smtids[1:]
            q_pos_decoder_input_ids = pos_smtids[:-1]
            q_neg_decoder_input_ids = neg_smtids[:-1]
        else:
            pos_doc_encoding = self.docid_to_smtid[str(positive)][1:]
            neg_doc_encoding = self.docid_to_smtid[str(negative)][1:]
            q_pos_decoder_input_ids = self.docid_to_smtid[str(positive)][:-1]
            q_neg_decoder_input_ids = self.docid_to_smtid[str(negative)][:-1]

        q_pos = "query: " + q.strip()
        q_neg = "query: " + q.strip()

        return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg, \
        q_pos_decoder_input_ids, q_neg_decoder_input_ids

class MarginMSEforPretrainDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir, qrels_path,
                 docid_to_smtid_path):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        if docid_to_smtid_path is not None:
            with open(docid_to_smtid_path) as fin: 
                self.docid_to_smtid = ujson.load(fin)
            tmp_docids = list(self.docid_to_smtid.keys())
            assert self.docid_to_smtid[tmp_docids[0]][0] == -1, self.docid_to_smtid[tmp_docids[0]]
        else:
            self.docid_to_smtid = None 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        positive = example["docids"][0]
        s_pos = example["scores"][0]

        neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
        negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]
        #assert negative != positive, (positive, negative)

        q = self.query_dataset[str(query)][1]
        d_pos = self.document_dataset[positive][1]
        d_neg = self.document_dataset[negative][1]

        if self.docid_to_smtid is not None:
            pos_prev_smtids = self.docid_to_smtid[str(positive)][1:]
            neg_prev_smtids = self.docid_to_smtid[str(negative)][1:]

            d_pos_decoder_input_ids = self.docid_to_smtid[str(positive)]
            d_neg_decoder_input_ids = self.docid_to_smtid[str(negative)]
            q_pos_decoder_input_ids = copy.deepcopy(d_pos_decoder_input_ids)
            q_neg_decoder_input_ids = copy.deepcopy(d_neg_decoder_input_ids)
        else:
            pos_prev_smtids = None 
            neg_prev_smtids = None
            d_pos_decoder_input_ids = [-1]
            d_neg_decoder_input_ids = [-1]
            q_pos_decoder_input_ids = [-1]
            q_neg_decoder_input_ids = [-1]

        q_pos = "query: " + q.strip()
        q_neg = "query: " + q.strip()
        d_pos = "document: " + d_pos.strip()
        d_neg = "document: " + d_neg.strip()

        if pos_prev_smtids is None and neg_prev_smtids is None:
            return q_pos, q_neg, d_pos, d_neg, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids 
        else:
            return q_pos, q_neg, d_pos, d_neg, pos_prev_smtids, neg_prev_smtids, s_pos, s_neg, \
            d_pos_decoder_input_ids, d_neg_decoder_input_ids, q_pos_decoder_input_ids, q_neg_decoder_input_ids