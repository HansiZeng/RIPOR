import os 
import json
import ujson
import time
import copy
import pickle

import faiss
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import HfArgumentParser
from transformers import T5ForConditionalGeneration
import torch
import numpy as np

from .dataset.dataloader import (
    CollectionDataLoader, 
    CollectionDataWithDocIDLoader, 
    CollectionAQLoader,
)
from .dataset.dataset import (
    CollectionDatasetPreLoad, 
    CollectionDatasetWithDocIDPreLoad, 
    CollectionAQDataset,
)
from .tasks.evaluator import DenseIndexing, DenseRetriever, AddictvieQuantizeIndexer
from .modeling.t5model_encoder import T5ModelEncoder
from .modeling.t5_generative_retriever import  (
    T5SeqPretrainEncoder,
    T5AQEncoder,
    T5SeqAQEncoder
    )

from .utils.utils import get_dataset_name, convert_ptsmtids_to_strsmtid, from_qrel_to_qsmtid_rel
from .utils.metrics import load_and_evaluate, load_and_evaluate_for_qid_smtid
from .arguments import ModelArguments, EvalArguments
from .tasks.generation import (
    generate_for_constrained_prefix_beam_search, 
    PrefixConstrainLogitProcessorFastSparse, 
    )
from .utils.metrics import mrr_k

def constrained_decode(model, 
                        dataloader,
                        #valid_smtids, 
                        prefix_constrain_processor,
                        smtid_to_docid,
                        max_new_token,
                        device, 
                        out_dir,
                        local_rank,
                        topk=100):
    
    qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}
            outputs = generate_for_constrained_prefix_beam_search(
                        model,
                        prefix_constrain_processor,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk
                    )
        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docid:
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    qid_to_rankdata[qid][smtid] = rel_score
                    
    out_path = os.path.join(out_dir, f"qid_to_smtid_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(qid_to_rankdata, fout)
    
def constrained_decode_doc(model, 
                        dataloader,
                        prefix_constrain_processor,
                        smtid_to_docids,
                        max_new_token,
                        device, 
                        out_dir,
                        local_rank,
                        topk=100,
                        apply_log_softmax_for_scores=False):
    
    qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}
            outputs = generate_for_constrained_prefix_beam_search(
                        model,
                        prefix_constrain_processor,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                        apply_log_softmax_for_scores=apply_log_softmax_for_scores
                    )
        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                if smtid not in smtid_to_docids:
                    print(f"smtid: {smtid} not in smtid_to_docid")
                else:
                    for docid in smtid_to_docids[smtid]:
                        if apply_log_softmax_for_scores:
                            qid_to_rankdata[qid][docid] = rel_score
                        else:
                            qid_to_rankdata[qid][docid] = rel_score * max_new_token
                    
    out_path = os.path.join(out_dir, f"run_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(qid_to_rankdata, fout)

def constrained_decode_smtid(model, 
                        dataloader,
                        prefix_constrain_processor,
                        smtid_to_docids,
                        max_new_token,
                        device, 
                        out_dir,
                        local_rank,
                        topk=100,
                        apply_log_softmax_for_scores=False):
    
    qid_to_rankdata = {}
    for i, batch in enumerate(tqdm(dataloader,total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}
            outputs = generate_for_constrained_prefix_beam_search(
                        model,
                        prefix_constrain_processor,
                        input_ids=inputs["input_ids"].long(),
                        attention_mask=inputs["attention_mask"].long(),
                        max_new_tokens=max_new_token,
                        output_scores=True,
                        return_dict=True,
                        return_dict_in_generate=True,
                        num_beams=topk,
                        num_return_sequences=topk,
                        apply_log_softmax_for_scores=apply_log_softmax_for_scores
                    )
        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(outputs.sequences.view(-1, topk, max_new_token+1), max_new_token)
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()
        for qid, ranked_smtids, rel_scores in zip(batch_qids, str_smtids, relevant_scores):
            qid_to_rankdata[qid] = {}
            for smtid, rel_score in zip(ranked_smtids, rel_scores):
                qid_to_rankdata[qid][smtid] = {}
                if smtid in smtid_to_docids:
                    for docid in smtid_to_docids[smtid]:
                        if apply_log_softmax_for_scores:
                            qid_to_rankdata[qid][smtid][docid] = rel_score
                        else:
                            qid_to_rankdata[qid][smtid][docid] = rel_score * max_new_token
                    
    out_path = os.path.join(out_dir, f"qid_smtid_rankdata_{local_rank}.json")
    with open(out_path, "w") as fout:
        ujson.dump(qid_to_rankdata, fout)


def ddp_setup():
    init_process_group(backend="nccl")

def index(args):
    ddp_setup()

    # parallel initialiaztion
    assert args.local_rank != -1
    if "t5_docid_gen_encoder" in args.pretrained_path:
        if args.encoder_type == "t5seq_pretrain_encoder":
            print("encoder_type: ", args.encoder_type)
            model = T5SeqPretrainEncoder.from_pretrained(args.pretrained_path)
        else:
            raise ValueError(f"{args.encoder_type} is not valid.")
    else:
        raise ValueError(f"{args.pretrained_path} is not supported.")
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if "t5_docid_gen_encoder" in args.pretrained_path:
        print("docid_to_smtid_path: {}".format(args.docid_to_smtid_path))
        d_collection = CollectionDatasetWithDocIDPreLoad(data_dir=args.collection_path, id_style="row_id", 
                                                         tid_to_smtid_path=args.docid_to_smtid_path,
                                                         add_prefix=True,
                                                        is_query=False)
        d_loader = CollectionDataWithDocIDLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                                max_length=args.max_length,
                                                batch_size=args.index_retrieve_batch_size,
                                                num_workers=1,
                                                sampler=DistributedSampler(d_collection, shuffle=False))
    elif "t5model_" in args.pretrained_path or "bert_" in args.pretrained_path:
        d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
        d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length,
                                        batch_size=args.index_retrieve_batch_size,
                                        num_workers=1,
                                    sampler=DistributedSampler(d_collection, shuffle=False))
        raise ValueError("this type of model not evaluted so far.")
    else:
        raise ValueError(f"The {args.pretrained_path} doesn't have existing dataset and dataloader")
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, args.local_rank, use_fp16=False)
    
    destroy_process_group()

def index_2(args):
    DenseIndexing.aggregate_embs_to_index(args.index_dir)

def retrieve(args):
    if "t5_docid_gen_encoder" in args.pretrained_path:
        if args.encoder_type == "t5seq_pretrain_encoder":
            print("encoder_type: ", args.encoder_type)
            model = T5SeqPretrainEncoder.from_pretrained(args.pretrained_path)
        else:
            raise ValueError(f"{args.encoder_type} is not valid.")
    else:
        raise ValueError(f"{args.pretrained_path} is not supported.")
    model.cuda()

    batch_size = 128

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        if "t5_docid_gen_encoder" in args.pretrained_path:
            q_collection = CollectionDatasetWithDocIDPreLoad(data_dir=data_dir, id_style="row_id",
                                                             tid_to_smtid_path=args.docid_to_smtid_path,
                                                             add_prefix=True,
                                                            is_query=True)
            q_loader = CollectionDataWithDocIDLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
        elif "t5model_" in args.pretrained_path or "bert_" in args.pretrained_path:
            q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
            q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                            max_length=args.max_length, batch_size=batch_size,
                                            shuffle=False, num_workers=4)
            raise ValueError("this type of model not evaluted so far.")
        else:
            raise ValueError(f"The {args.pretrained_path} doesn't have existing dataset and dataloader")
        evaluator = DenseRetriever(model=model, args=args, dataset_name=get_dataset_name(data_dir))
        evaluator.retrieve(q_loader, 
                        topk=args.topk)
    evaluate(args)

def evaluate(args):
    if len(args.eval_qrel_path) == 1:
        args.eval_qrel_path = ujson.loads(args.eval_qrel_path[0])
    eval_qrel_path = args.eval_qrel_path
    eval_metric = args.eval_metric
    out_dir = args.out_dir

    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics) in enumerate(zip(eval_qrel_path, eval_metric)):
        if qrel_file_path is not None:
            res = {}
            dataset_name = get_dataset_name(qrel_file_path)
            print(eval_metrics)
            for metric in eval_metrics:
                res.update(load_and_evaluate(qrel_file_path=qrel_file_path,
                                             run_file_path=os.path.join(out_dir, dataset_name, "run.json"),
                                             metric=metric))
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res
            json.dump(res, open(os.path.join(out_dir, dataset_name, "perf.json"), "a"))
    json.dump(res_all_datasets, open(os.path.join(out_dir, "perf_all_datasets.json"), "a"))
    return res_all_datasets

def mmap_2(args):
    DenseIndexing.aggregate_embs_to_mmap(args.mmap_dir)

def aq_index(args):
    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)
    AddictvieQuantizeIndexer.index(args.mmap_dir, index_dir=args.index_dir, num_subvectors_for_pq=args.num_subvectors_for_pq,
                                   codebook_bits=args.codebook_bits)

def aq_evaluate(args):
    model = T5AQEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    aq_indexer = AddictvieQuantizeIndexer(model, args)
    batch_size = 128

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetWithDocIDPreLoad(data_dir=data_dir, id_style="row_id",
                                                            tid_to_smtid_path=None,
                                                            add_prefix=True,
                                                            is_query=True)
        q_loader = CollectionDataWithDocIDLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
    
        aq_indexer.search(q_loader, 
                          topk=200, 
                          index_path=os.path.join(args.index_dir, "model.index"),
                          out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)),
                          index_ids_path=os.path.join(args.mmap_dir, "text_ids.tsv")
                          )

    evaluate(args)

def aq_to_flat_index_search_evaluate(args):
    """ this method is mainly for sanity check """
    # flat index
    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)
    index_path = os.path.join(args.index_dir, "model.index")

    model = T5AQEncoder.from_pretrained(args.pretrained_path)
    model.eval()
    model.cuda()

    print(args.collection_path, args.docid_to_smtid_path)
    dataset = CollectionAQDataset(args.collection_path, args.docid_to_smtid_path)
    dataloader = CollectionAQLoader(dataset=dataset, tokenizer_type=args.pretrained_path, max_length=128,
                                    batch_size=512, shuffle=False, num_workers=4)
    
    all_docids = []
    all_doc_reps = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        doc_encodings = batch["doc_encodings"].cuda()
        with torch.no_grad():
            doc_reps = model.decode(doc_encodings).cpu().numpy()
        
        all_doc_reps.append(doc_reps)
        all_docids.extend([int(x) for x in batch["docids"]])
    
    all_doc_reps = np.concatenate(all_doc_reps)
    all_docids = np.array(all_docids)

    index = faiss.IndexFlatIP(all_doc_reps.shape[1])
    index = faiss.IndexIDMap(index)

    index.add_with_ids(all_doc_reps, all_docids)
    faiss.write_index(index, index_path)

    # flat search
    batch_size = 128
    rq_indexer = AddictvieQuantizeIndexer(model, args)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
        
    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetWithDocIDPreLoad(data_dir=data_dir, id_style="row_id",
                                                            tid_to_smtid_path=None,
                                                            add_prefix=True,
                                                            is_query=True)
        q_loader = CollectionDataWithDocIDLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
        
        rq_indexer.flat_index_search(q_loader, 
                                     topk=200, 
                                     index_path=index_path, 
                                     out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)))
    # evaluate
    evaluate(args)

def t5seq_aq_retrieve_docids(args):
    ddp_setup()
    model = T5SeqAQEncoder.from_pretrained(args.pretrained_path)
    model.eval()
    with open(args.docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    print(args.docid_to_smtid_path)
    list_smtid_to_nextids_path = os.path.join(os.path.dirname(args.docid_to_smtid_path), "list_smtid_to_nextids.pkl")
    if "experiments-full" in args.docid_to_smtid_path and os.path.exists(list_smtid_to_nextids_path):
            print("read list_smtid_to_nextids from {}".format(list_smtid_to_nextids_path))
            with open(list_smtid_to_nextids_path, "rb") as fin:
                list_smtid_to_nextids = pickle.load(fin)
    else:
        # create prefix_constrain logit processor
        list_smtid_to_nextids = [dict() for _ in range(len(docid_to_smtids["0"])-1)]
        for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
            for i in range(len(smtids)-1):
                cur_smtid = "_".join([str(x) for x in smtids[:i+1]])
                next_id = int(smtids[i+1])

                cur_dict = list_smtid_to_nextids[i]
                if cur_smtid not in cur_dict:
                    cur_dict[cur_smtid] = {next_id}
                else:
                    cur_dict[cur_smtid].add(next_id)
        for i, cur_dict in enumerate(list_smtid_to_nextids):
            for cur_smtid in cur_dict:
                cur_dict[cur_smtid] = list(cur_dict[cur_smtid])
            if args.local_rank <= 0:
                print(f"{i}-th step has {len(cur_dict):,} effective smtid ")

        if "experiments-full" in args.docid_to_smtid_path:
            if args.local_rank <= 0:
                print("save list_smtid_to_nextids")
                with open(list_smtid_to_nextids_path, "wb") as fout:
                    pickle.dump(list_smtid_to_nextids, fout)
    if len(set(model.config.decoder_vocab_sizes)) == 1:
        prefix_constrain_processor = PrefixConstrainLogitProcessorFastSparse(list_smtid_to_nextids, model.config.decoder_vocab_sizes[0])
    else:
        raise ValueError("not valid decoder_vocab_size")

    # define parameters for decoding
    smtid_to_docids = {}
    for docid, smtids in docid_to_smtids.items():
        assert smtids[0] == -1, smtids
        sid = "_".join([str(x) for x in smtids[1:1+args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if args.local_rank <= 0:
        print("max_new_token: ", max_new_token)
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    # start to retrieve
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        dev_dataset = CollectionDatasetWithDocIDPreLoad(data_dir=data_dir, id_style="row_id",
                                              tid_to_smtid_path=None, add_prefix=True, is_query=True)
        dev_loader =  CollectionDataWithDocIDLoader(dataset=dev_dataset, 
                                                    tokenizer_type=args.pretrained_path,
                                                    max_length=256,
                                                    batch_size=args.batch_size,
                                                    num_workers=1,
                                                    sampler=DistributedSampler(dev_dataset, shuffle=False))

        model.to(args.local_rank)
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir)
        if args.local_rank <= 0:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        model.base_model.config.decoding = True
        constrained_decode_doc(model.base_model, 
                            dev_loader,
                            prefix_constrain_processor,
                            smtid_to_docids,
                            max_new_token,
                            device=args.local_rank, 
                            out_dir=out_dir,
                            local_rank=args.local_rank,
                            topk=args.topk,
                            apply_log_softmax_for_scores=args.apply_log_softmax_for_scores)

def t5seq_aq_retrieve_docids_2(args):
    if len(args.q_collection_paths) == 1:
        args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
        print(args.q_collection_paths)
    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old 
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))
        
        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
                sub_qid_to_rankdata = ujson.load(fin)
            if len(qid_to_rankdata) == 0:
                qid_to_rankdata.update(sub_qid_to_rankdata)
            else:
                for qid, rankdata in sub_qid_to_rankdata.items():
                    if qid not in qid_to_rankdata:
                        qid_to_rankdata[qid] = rankdata
                    else:
                        qid_to_rankdata[qid].update(rankdata)
        print("length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

    evaluate(args)

def t5seq_aq_get_qid_to_smtid_rankdata(args):
    ddp_setup()
    model = T5SeqAQEncoder.from_pretrained(args.pretrained_path)
    model.eval()

    with open(args.docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    # create prefix_constrain logit processor
    list_smtid_to_nextids = [dict() for _ in range(len(docid_to_smtids["0"])-1)]
    for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
        for i in range(len(smtids)-1):
            cur_smtid = "_".join([str(x) for x in smtids[:i+1]])
            next_id = int(smtids[i+1])

            cur_dict = list_smtid_to_nextids[i]
            if cur_smtid not in cur_dict:
                cur_dict[cur_smtid] = {next_id}
            else:
                cur_dict[cur_smtid].add(next_id)
    for i, cur_dict in enumerate(list_smtid_to_nextids):
        for cur_smtid in cur_dict:
            cur_dict[cur_smtid] = list(cur_dict[cur_smtid])
        if args.local_rank <= 0:
            print(f"{i}-th step has {len(cur_dict):,} effective smtid ")

    #prefix_constrain_processor = PrefixConstrainLogitProcessorFastSparse(list_smtid_to_nextids, model.config.decoder_vocab_sizes[0])
    if len(set(model.config.decoder_vocab_sizes)) == 1:
            prefix_constrain_processor = PrefixConstrainLogitProcessorFastSparse(list_smtid_to_nextids, model.config.decoder_vocab_sizes[0])
    elif len(set(model.config.decoder_vocab_sizes)) == 2:
        raise NotImplementedError
        #print("the decoder_vocab_sizes are not same, hence use multishape PrefixConstrain")
        #prefix_constrain_processor = PrefixConstrainLogitProcessorFastSparseMultiShape(list_smtid_to_nextids, 
        #                                                                                model.config.decoder_vocab_sizes)
    else:
        raise ValueError("not valid decoder_vocab_size")

    # define parameters for decoding
    assert args.max_new_token in [4, 8, 16, 32], args.max_new_token
    smtid_to_docids = {}
    for docid, smtids in docid_to_smtids.items():
        assert smtids[0] == -1, smtids
        sid = "_".join([str(x) for x in smtids[1:1+args.max_new_token]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]
    if args.local_rank <= 0:
        print("distribution of docids length per smtid: ", 
              np.quantile([len(xs) for xs in smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
        print("avergen length = {:.3f}".format(np.mean([len(xs) for xs in smtid_to_docids.values()])))
        print("smtid: ", sid)

    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
    
    dev_dataset = CollectionDatasetWithDocIDPreLoad(data_dir=args.train_query_dir, id_style="row_id",
                                            tid_to_smtid_path=None, add_prefix=True, is_query=True)
    dev_loader =  CollectionDataWithDocIDLoader(dataset=dev_dataset, 
                                                tokenizer_type=args.pretrained_path,
                                                max_length=256,
                                                batch_size=args.batch_size,
                                                num_workers=1,
                                                sampler=DistributedSampler(dev_dataset, shuffle=False))

    model.to(args.local_rank)
    out_dir = args.out_dir
    print("out_dir: ", out_dir)
    if args.local_rank <= 0:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    model.base_model.config.decoding = True
    constrained_decode_smtid(model.base_model, 
                        dev_loader,
                        prefix_constrain_processor,
                        smtid_to_docids,
                        args.max_new_token,
                        device=args.local_rank, 
                        out_dir=out_dir,
                        local_rank=args.local_rank,
                        topk=args.topk,
                        apply_log_softmax_for_scores=args.apply_log_softmax_for_scores)
        
def t5seq_aq_get_qid_to_smtid_rankdata_2(args):
    out_dir = args.out_dir

    # remove old 
    if os.path.exists(os.path.join(out_dir, "qid_smtid_rankdata.json")):
        print("old run.json exisit.")
        os.remove(os.path.join(out_dir, "qid_smtid_rankdata.json"))
    
    # merge
    qid_to_smtid_rankdata = {}
    sub_paths = [p for p in os.listdir(out_dir) if "qid_smtid_rankdata" in p]
    assert len(sub_paths) == torch.cuda.device_count()
    for sub_path in sub_paths:
        with open(os.path.join(out_dir, sub_path)) as fin:
            sub_qid_to_smtid_rankdata = ujson.load(fin)
        for qid in sub_qid_to_smtid_rankdata:
            if qid not in qid_to_smtid_rankdata:
                qid_to_smtid_rankdata[qid] = sub_qid_to_smtid_rankdata[qid]
            else:
                for smtid in sub_qid_to_smtid_rankdata[qid]:
                    if smtid not in qid_to_smtid_rankdata[qid]:
                        qid_to_smtid_rankdata[qid][smtid] = sub_qid_to_smtid_rankdata[qid][smtid]
                    else:
                        for docid, score in sub_qid_to_smtid_rankdata[qid][smtid].items():
                            qid_to_smtid_rankdata[qid][smtid][docid] = score
    
    smtid_lengths = []
    doc_lenghts = []
    for qid in qid_to_smtid_rankdata:
        smtid_lengths.append(len(qid_to_smtid_rankdata[qid]))
        for smtid in qid_to_smtid_rankdata[qid]:
            doc_lenghts.append(len(qid_to_smtid_rankdata[qid][smtid]))
    
    print("smtid_length per query: ", np.quantile(smtid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("doc_length per smtid: ", np.quantile(doc_lenghts, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))


    with open(os.path.join(out_dir, "qid_smtid_rankdata.json"), "w") as fout:
        ujson.dump(qid_to_smtid_rankdata, fout)

    for sub_path in sub_paths:
        sub_path = os.path.join(out_dir, sub_path)
        os.remove(sub_path)

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.task == "index":
        index(args)
    elif args.task == "index_2":
        index_2(args)
    elif args.task == "retrieve":
        retrieve(args)
    elif args.task == "evaluate":
        evaluate(args)
    elif args.task == "mmap":
        assert "mmap" in args.index_dir, args.index_dir
        index(args)
    elif args.task == "mmap_2":
        assert args.mmap_dir == args.index_dir, (args.mmap_dir, args.index_dir)
        mmap_2(args)
    elif args.task == "aq_index":
        aq_index(args)
    elif args.task == "aq_evaluate":
        aq_evaluate(args)
    elif args.task == "aq_to_flat_index_search_evaluate":
        aq_to_flat_index_search_evaluate(args)
    elif args.task == "t5seq_aq_retrieve_docids":
        t5seq_aq_retrieve_docids(args)
    elif args.task == "t5seq_aq_retrieve_docids_2":
        t5seq_aq_retrieve_docids_2(args)
    elif args.task == "t5seq_aq_get_qid_to_smtid_rankdata":
        t5seq_aq_get_qid_to_smtid_rankdata(args)
    elif args.task == "t5seq_aq_get_qid_to_smtid_rankdata_2":
        t5seq_aq_get_qid_to_smtid_rankdata_2(args)
    else:
        raise ValueError(f"task: {args.task} is not valid.")
