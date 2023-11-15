import os 
import ujson 
import gzip
import pickle
import random

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_world_size
from torch.utils.data.distributed import DistributedSampler
import torch
from transformers import HfArgumentParser
import numpy as np

from .modeling.cross_encoder import CrossEncoder
from .dataset.dataset import (
    PseudoQueryForScoreDataset, 
    QueryToSmtidRerankDataset, 
    TeacherRerankFromQidSmtidsDataset,
    CrossEncRerankForSamePrefixPair,
    RerankDataset
)
from .dataset.dataloader import (
    CrossEncRerankDataLoader, 
    PseudoQueryForScoreDataLoader, 
    QueryToSmtidRerankLoader,
    CrossEncRerankForSamePrefixPairLoader
    )
from .tasks.reranker import Reranker
from .arguments import RerankArguments
from .utils.metrics import mrr_k, load_and_evaluate
from .utils.utils import sample_from_partitions, get_dataset_name

random.seed(4)

def ddp_setup():
    init_process_group(backend="nccl")

def rerank_for_eval(args):
    raise NotImplementedError 

def rerank_for_create_trainset(args):
    ddp_setup()
    print("model_name_or_path: ", args.model_name_or_path)
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    #assert len(args.run_json_paths) == 1 and len(args.q_collection_paths) == 1
    run_json_path = args.run_json_path
    q_collection_path = args.q_collection_path
    #assert os.path.dirname(run_json_path) == args.out_dir, (run_json_path, args.out_dir)

    rerank_dataset = RerankDataset(run_json_path=run_json_path, document_dir=args.collection_path, query_dir=q_collection_path,
                                   json_type=args.json_type)
    rerank_loader = CrossEncRerankDataLoader(dataset=rerank_dataset, 
                                    tokenizer_type=args.model_name_or_path,
                                    max_length=args.max_length, 
                                    batch_size=args.batch_size,
                                    shuffle=False, num_workers=1,
                                sampler=DistributedSampler(rerank_dataset) if args.local_rank != -1 else None)
    reranker = Reranker(model=model, dataloader=rerank_loader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    reranker.reranking(name=f"{args.local_rank}", use_fp16=True)

def rerank_for_create_trainset_2(args):
    if os.path.exists(os.path.join(args.out_dir, "qid_pids_rerank_scores.train.json")):
        print("old qid_pids_rerank_scores.train.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_pids_rerank_scores.train.json"))

    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count(), (len(sub_rerank_paths), torch.cuda.device_count())
    qid_to_rankdata = {}
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)
    print("length of qids and avg rankdata length in qid_to_rankdata: {}, {}".format(
    len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

    qid_to_sorteddata = {}
    for qid, rankdata in qid_to_rankdata.items():
        qid_to_sorteddata[qid] = dict(sorted(rankdata.items(), key=lambda x: x[1], reverse=True))
    
    with open(os.path.join(args.out_dir, "qid_docids_teacher_scores.train.json"), "w") as fout:
        for qid, rankdata in qid_to_sorteddata.items():
            example = {"qid": qid, "docids": [], "scores": []}
            for i, (pid, score) in enumerate(rankdata.items()):
                if i == 200:
                    break
                example["docids"].append(pid)
                example["scores"].append(score)
            fout.write(ujson.dumps(example) + "\n")
    print("end write to json")

    for sub_path in sub_rerank_paths:
        os.remove(os.path.join(args.out_dir, sub_path))

    #other_qid_to_data = {}
    #for qid, rankdata in qid_to_sorteddata.items():
    #    other_qid_to_data[int(qid)] = {int(k): v for k,v in rankdata.items()}
    #with gzip.open(os.path.join(args.out_dir, "teacher_score.pkl.gz"), "wb") as fout:
    #    pickle.dump(other_qid_to_data, fout)
    #print("end write to pickle")

def rerank_for_evaluate_2(args):
    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
    if os.path.exists(os.path.join(args.out_dir, "qid_to_rerank_data.json")):
        print("old qid_pids_rerank_scores.train.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_to_rerank_data.json"))

    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    qid_to_rankdata = {}
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)

    print("length of qids and avg rankdata length in qid_to_rankdata: {}, {}".format(
    len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

    with open(os.path.join(args.out_dir, "qid_to_rerank_data.json"), "w") as fout:
        ujson.dump(qid_to_rankdata, fout)

    for sub_path in sub_rerank_paths:
        os.remove(os.path.join(args.out_dir, sub_path))

    # evaluate
    res = {}
    args.eval_metrics = ujson.loads(args.eval_metrics)
    print(args.eval_metrics)
    for metric in args.eval_metrics:
        res.update(load_and_evaluate(qrel_file_path=args.eval_qrel_path,
                                    run_file_path=os.path.join(args.out_dir, "qid_to_rerank_data.json"),
                                    metric=metric))
            
    ujson.dump(res, open(os.path.join(args.out_dir, "rerank_perf.json"), "w"))
    print(res)

    
def assign_scores_for_pseudo_queries(args):
    ddp_setup()
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    dataset = PseudoQueryForScoreDataset(document_dir=args.collection_path, pseudo_queries_path=args.pseudo_queries_path, 
                                        docid_pseudo_qids_path=args.docid_pseudo_qids_path)
    dataloader = PseudoQueryForScoreDataLoader(dataset=dataset, 
                                    tokenizer_type=args.model_name_or_path,
                                    max_length=args.max_length, 
                                    batch_size=args.batch_size,
                                    shuffle=False, num_workers=1,
                                sampler=DistributedSampler(dataset) if args.local_rank != -1 else None)
    reranker = Reranker(model=model, dataloader=dataloader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    #partition = args.docid_pseudo_qids_path.split(".")[0].split("_")[-1]
    #reranker.assign_scores_for_pseudo_queries(name=f"{partition}_{args.local_rank}", use_fp16=True)
    reranker.assign_scores_for_pseudo_queries(name=f"{args.local_rank}", use_fp16=True)

def assign_scores_for_pseudo_queries_2(args):
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "pid_qids_rerank_scores" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    qid_to_rankdata = {}
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)
    print("length of qids and avg rankdata length in pid_to_rankdata: {}, {}".format(
    len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

    with open(os.path.join(args.out_dir, "pid_qids_rerank_scores.json"), "w") as fout:
        ujson.dump(qid_to_rankdata, fout)


def query_to_docid_rerank_for_qid_smtids(args):
    ddp_setup()
    with open(args.docid_to_smtid_path) as fin:
        docid_to_smtid = ujson.load(fin)

    docid_to_strsmtid = {}
    for docid, smtid in docid_to_smtid.items():
        assert smtid[0] == -1, smtid 
        str_smtid = "_".join([str(x) for x in smtid[1:]])
        docid_to_strsmtid[docid] = str_smtid 
    
    # dataset and dataloader 
    dataset = QueryToSmtidRerankDataset(qid_docids_path=args.qid_docids_path,
                                        queries_path=args.dev_queries_path,
                                        docid_to_smtids=docid_to_smtid,
                                        docid_to_strsmtid=docid_to_strsmtid)
    dataloader = QueryToSmtidRerankLoader(dataset=dataset, 
                                    tokenizer_type=args.query_to_smtid_tokenizer_type,
                                    max_length=args.max_length, 
                                    batch_size=args.batch_size,
                                    shuffle=False, num_workers=1,
                                    sampler=DistributedSampler(dataset) if args.local_rank != -1 else None)
    # debug 
    """
    for i, batch in enumerate(dataloader):
        print("query: ")
        print(dataloader.tokenizer.batch_decode((batch["tokenized_query"]["input_ids"][:8])))
        print("-"*100)
        print("decoder_input_ids: ")
        print(batch["tokenized_query"]["decoder_input_ids"][:8])
        print("-"*100)
        print("pair_ids: ")
        print(batch["pair_ids"][:8])
        print("-"*100)
        print("labels: ")
        print(batch["labels"][:8])
        if i == 1:
            break
        print("="*100)
    exit()
    """
        
    
    model = T5SeqQueryToDocidEncoder.from_pretrained(args.pretrained_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    reranker = Reranker(model=model, dataloader=dataloader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    
    reranker.query_to_smtid_reranking(name=f"{args.local_rank}", use_fp16=False)

def query_to_docid_rerank_for_qid_smtids_2(args):
    if os.path.exists(os.path.join(args.out_dir, "qid_smtids_rerank.json")):
        print("old qid_smtids_rerank.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_smtids_rerank.json"))

    qid_to_rankdata = {}
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "qid_smtids_rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)
    print("length of pids and avg rankdata length in pid_to_rankdata: {}, {}".format(
    len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

    with open(os.path.join(args.out_dir, "qid_smtids_rerank.json"), "w") as fout:
        ujson.dump(qid_to_rankdata, fout)

    for sub_path in sub_rerank_paths:
        sub_path = os.path.join(args.out_dir, sub_path)
        os.remove(sub_path)

    with open(args.docid_to_smtid_path) as fin:
        docid_to_smtid = ujson.load(fin)

    docid_to_strsmtid = {}
    for docid, smtid in docid_to_smtid.items():
        assert smtid[0] == -1, smtid 
        str_smtid = "_".join([str(x) for x in smtid[1:]])
        docid_to_strsmtid[docid] = str_smtid 

    with open(args.dev_qrels_path) as fin:
        qrel_data = ujson.load(fin)
    qid_to_relsmtid_data = {}
    for qid in qrel_data:
        qid_to_relsmtid_data[qid] = {}
        for docid, s in qrel_data[qid].items():
            rel_smtid = docid_to_strsmtid[docid]
            qid_to_relsmtid_data[qid][rel_smtid] = s 

    mrr_at_10 = mrr_k(run=qid_to_rankdata, qrel=qid_to_relsmtid_data, k=10)
    mrr_at_100 = mrr_k(run=qid_to_rankdata, qrel=qid_to_relsmtid_data, k=100)

    with open(os.path.join(args.out_dir, "metric.json"), "w") as fout:
        ujson.dump({
            "mrr_at_10": mrr_at_10,
            "mrr_at_100": mrr_at_100
        }, fout)

def teacher_rerank_for_qid_smtids(args):
    ddp_setup()
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    rerank_dataset = TeacherRerankFromQidSmtidsDataset(
        qid_smtid_rank_path=args.qid_smtid_rank_path,
        docid_to_smtids_path=args.docid_to_smtid_path,
        queries_path=args.dev_queries_path,
        collection_path=os.path.join(args.collection_path, "raw.tsv")
    )
    rerank_loader = CrossEncRerankDataLoader(dataset=rerank_dataset, 
                                    tokenizer_type=args.model_name_or_path,
                                    max_length=args.max_length, 
                                    batch_size=args.batch_size,
                                    shuffle=False, num_workers=1,
                                sampler=DistributedSampler(rerank_dataset) if args.local_rank != -1 else None)

    assert args.out_dir in args.qid_smtid_rank_path, (args.out_dir, args.qid_smtid_rank_path)
    reranker = Reranker(model=model, dataloader=rerank_loader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    reranker.reranking(name=f"teacher_{args.local_rank}", use_fp16=True)

def teacher_rerank_for_qid_smtids_2(args):
    if os.path.exists(os.path.join(args.out_dir, "rerank_teacher.json")):
        print("old rerank_teacher.json exisit.")
        os.remove(os.path.join(args.out_dir, "rerank_teacher.json"))

    qid_to_rankdata = {}
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "rerank_teacher" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)
    print("length of pids and avg rankdata length in pid_to_rankdata: {}, {}".format(
    len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])))

    with open(os.path.join(args.out_dir, "rerank_teacher.json"), "w") as fout:
        ujson.dump(qid_to_rankdata, fout)

    for sub_path in sub_rerank_paths:
        sub_path = os.path.join(args.out_dir, sub_path)
        os.remove(sub_path)

def cross_encoder_rerank_for_same_prefix_docid(args):
    ddp_setup()
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    with open(args.docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    docid_to_smtid = {}
    smtid_to_docids = {}
    for docid, smtids in docid_to_smtids.items():
        assert smtids[0] == -1, smtids
        sid = "_".join([str(x) for x in smtids[1:]])

        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

        docid_to_smtid[docid] = sid 
    docid_to_smtids = None

    qid_to_reldocids = {}
    with open(args.train_qrels_path) as fin:
        train_qrels_data = ujson.load(fin)
    for qid, qrel_data in train_qrels_data.items():
        for docid in qrel_data:
            if qid not in qid_to_reldocids:
                qid_to_reldocids[qid] = [docid]
            else:
                qid_to_reldocids[qid] += [docid]

    qid_to_smtid_to_reldocids = {}
    qid_to_smtid_to_docids = {}
    for i, qid in enumerate(qid_to_reldocids):
        qid_to_smtid_to_reldocids[qid] = {}
        if i % get_world_size() == args.local_rank:
            qid_to_smtid_to_docids[qid] = {}
        for reldocid in qid_to_reldocids[qid]:
            smtid = docid_to_smtid[reldocid]
            if smtid not in qid_to_smtid_to_reldocids[qid]:
                qid_to_smtid_to_reldocids[qid] = {smtid: [reldocid]}
            else:
                qid_to_smtid_to_reldocids[qid][smtid].append(reldocid)

            if i % get_world_size() == args.local_rank:
                neg_docids = random.sample(smtid_to_docids[smtid], k=min(50, len(smtid_to_docids[smtid])))
                if smtid not in qid_to_smtid_to_docids[qid]:
                    qid_to_smtid_to_docids[qid] = {smtid: neg_docids} #{smtid: smtid_to_docids[smtid]}
                else:
                    qid_to_smtid_to_docids[qid][smtid] = neg_docids #smtid_to_docids[smtid]

    print("number of gpus = {}, current rank = {}".format(get_world_size(), args.local_rank))
    print("length of qid_to_smtid_to_reldocids = {}, qid_to_smtid_to_docids = {}, ratio = {:.3f}".format(
        len(qid_to_smtid_to_reldocids), len(qid_to_smtid_to_docids), len(qid_to_smtid_to_reldocids) / len(qid_to_smtid_to_docids)
    ))
    rerank_dataset = CrossEncRerankForSamePrefixPair(qid_to_smtid_to_docids,
                                              queries_path=args.train_queries_path,
                                              collection_path=os.path.join(args.collection_path, "raw.tsv"))
    rerank_loader = CrossEncRerankForSamePrefixPairLoader(dataset=rerank_dataset, 
                                            tokenizer_type=args.model_name_or_path,
                                            max_length=args.max_length, 
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1) # we sub-sample qid_to_reldocids here, so we don't need DistributedSampler
                                        #sampler=DistributedSampler(rerank_dataset) if args.local_rank != -1 else None)
    reranker = Reranker(model=model, dataloader=rerank_loader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    reranker.reranking_for_same_prefix_pair(name=f"{args.local_rank}", use_fp16=True)

def cross_encoder_rerank_for_same_prefix_docid_2(args):
    if os.path.exists(os.path.join(args.out_dir, "qid_to_smtid_to_rerank.json")):
        print("old qid_to_smtid_to_rerank.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_to_smtid_to_rerank.json"))

    qid_to_smtid_to_rerank = {}
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "qid_to_smtid_to_rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rerank = ujson.load(fin)
        
        for qid in sub_qid_to_rerank:
            for smtid in sub_qid_to_rerank[qid]:
                if qid not in qid_to_smtid_to_rerank:
                    qid_to_smtid_to_rerank[qid] = {smtid: sub_qid_to_rerank[qid][smtid]}
                else:
                    if smtid not in qid_to_smtid_to_rerank[qid]:
                        qid_to_smtid_to_rerank[qid][smtid] = sub_qid_to_rerank[qid][smtid]
                    else:
                        qid_to_smtid_to_rerank[qid][smtid] += sub_qid_to_rerank[qid][smtid]

    sorted_rerank = {}
    sorted_sample_rerank = {}
    lenghts = []
    num_pair = 0
    #sample_num = 50
    for qid in qid_to_smtid_to_rerank:
        sorted_rerank[qid] = {}
        sorted_sample_rerank[qid] = {}
        for smtid in qid_to_smtid_to_rerank[qid]:
            rerank_data = sorted(qid_to_smtid_to_rerank[qid][smtid], key=lambda x: x[1], reverse=True)

            sorted_rerank[qid][smtid] = rerank_data
            lenghts.append(len(rerank_data))
            num_pair += 1

            #if len(rerank_data) > sample_num:
            #    sorted_sample_rerank[qid][smtid] = sorted(sample_from_partitions(rerank_data, num_partitions=20, num_samples=sample_num),
            #                                               key=lambda x: x[1], reverse=True)
            #else:
            sorted_sample_rerank[qid][smtid] = rerank_data
        
    print("number of qid_smtid example = {}, average length of docs for each pair = {}".format(num_pair, np.mean(lenghts)))

    with open(os.path.join(args.out_dir, "qid_to_smtid_to_rerank.json"), "w") as fout:
        ujson.dump(sorted_rerank, fout)

    with open(os.path.join(args.out_dir, "qid_to_smtid_to_sampled_rerank.json"), "w") as fout:
        ujson.dump(sorted_sample_rerank, fout)

    for sub_path in sub_rerank_paths:
        sub_path = os.path.join(args.out_dir, sub_path)
        os.remove(sub_path)

def cross_encoder_rerank_for_same_reldocid_hard_docids(args):
    ddp_setup()
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.local_rank <= 0:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    with open(args.qid_to_reldocid_hard_docids_path) as fin:
        qid_to_reldocid_hard_docids = ujson.load(fin)
    
    sampled_data = {}
    for i, qid in enumerate(qid_to_reldocid_hard_docids):
        if i % get_world_size() == args.local_rank:
            sampled_data[qid] = qid_to_reldocid_hard_docids[qid]
    print("size of sampled_data = {}, original data = {}, ratio = {:.3f}".format(
        len(sampled_data), len(qid_to_reldocid_hard_docids), len(qid_to_reldocid_hard_docids) / len(sampled_data)
    ))
    rerank_dataset = CrossEncRerankForSamePrefixPair(sampled_data,
                                              queries_path=args.train_queries_path,
                                              collection_path=os.path.join(args.collection_path, "raw.tsv"))
    rerank_loader = CrossEncRerankForSamePrefixPairLoader(dataset=rerank_dataset, 
                                            tokenizer_type=args.model_name_or_path,
                                            max_length=args.max_length, 
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1)
                                            #sampler=DistributedSampler(rerank_dataset) if args.local_rank != -1 else None)
    reranker = Reranker(model=model, dataloader=rerank_loader, 
                        config={"out_dir": args.out_dir},
                        write_to_disk=True,
                        local_rank=args.local_rank)
    reranker.reranking_for_same_prefix_pair(name=f"{args.local_rank}", use_fp16=True, prefix_name="qid_to_reldocid_to_hard_rerank")

def cross_encoder_rerank_for_same_reldocid_hard_docids_2(args):
    if os.path.exists(os.path.join(args.out_dir, "qid_to_reldocid_to_hard_rerank.json")):
        print("old qid_to_reldocid_to_hard_rerank.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_to_reldocid_to_hard_rerank.json"))

    qid_to_smtid_to_rerank = {}
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "qid_to_reldocid_to_hard_rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rerank = ujson.load(fin)
        
        for qid in sub_qid_to_rerank:
            for smtid in sub_qid_to_rerank[qid]:
                if qid not in qid_to_smtid_to_rerank:
                    qid_to_smtid_to_rerank[qid] = {smtid: sub_qid_to_rerank[qid][smtid]}
                else:
                    if smtid not in qid_to_smtid_to_rerank[qid]:
                        qid_to_smtid_to_rerank[qid][smtid] = sub_qid_to_rerank[qid][smtid]
                    else:
                        qid_to_smtid_to_rerank[qid][smtid] += sub_qid_to_rerank[qid][smtid]

    sorted_rerank = {}
    sorted_sample_rerank = {}
    lenghts = []
    num_pair = 0
    sample_num = 200
    for qid in qid_to_smtid_to_rerank:
        sorted_rerank[qid] = {}
        sorted_sample_rerank[qid] = {}
        for smtid in qid_to_smtid_to_rerank[qid]:
            rerank_data = sorted(qid_to_smtid_to_rerank[qid][smtid], key=lambda x: x[1], reverse=True)

            sorted_rerank[qid][smtid] = rerank_data
            lenghts.append(len(rerank_data))
            num_pair += 1

            if len(rerank_data) > sample_num:
                sorted_sample_rerank[qid][smtid] = sorted(sample_from_partitions(rerank_data, num_partitions=20, num_samples=sample_num),
                                                           key=lambda x: x[1], reverse=True)
        
    print("number of qid_smtid example = {}, average length of docs for each pair = {}".format(num_pair, np.mean(lenghts)))

    with open(os.path.join(args.out_dir, "qid_to_reldocid_to_hard_rerank.json"), "w") as fout:
        ujson.dump(sorted_rerank, fout)

    with open(os.path.join(args.out_dir, "qid_to_reldocid_to_sampled_hard_rerank.json"), "w") as fout:
        ujson.dump(sorted_sample_rerank, fout)

    for sub_path in sub_rerank_paths:
        sub_path = os.path.join(args.out_dir, sub_path)
        os.remove(sub_path)
    
def cross_encoder_rerank_for_qid_smtid_docids(args):
    ddp_setup()
    print("model_name_or_path for cross_encoder: ", args.model_name_or_path)
    model = CrossEncoder(args.model_name_or_path)
    model.to(args.local_rank)

    with open(args.qid_smtid_docids_path) as fin:
        qid_to_smtid_to_docids = ujson.load(fin)
    print("qid_smtid_docids_path: ", args.qid_smtid_docids_path)

    sampled_data = {}
    for i, qid in enumerate(qid_to_smtid_to_docids):
        if i % get_world_size() == args.local_rank:
            sampled_data[qid] = qid_to_smtid_to_docids[qid]
    print("size of sampled_data = {}, original data = {}, ratio = {:.3f}".format(
        len(sampled_data), len(qid_to_smtid_to_docids), len(qid_to_smtid_to_docids) / len(sampled_data)
    ))

    rerank_dataset = CrossEncRerankForSamePrefixPair(sampled_data,
                                              queries_path=args.train_queries_path,
                                              collection_path=os.path.join(args.collection_path, "raw.tsv"))
    rerank_loader = CrossEncRerankForSamePrefixPairLoader(dataset=rerank_dataset, 
                                            tokenizer_type=args.model_name_or_path,
                                            max_length=args.max_length, 
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=1)
    reranker = Reranker(model=model, dataloader=rerank_loader, 
                        config=None,
                        local_rank=args.local_rank,
                        write_to_disk=False)
    
    qid_to_smtid_to_rankdata = reranker.reranking_for_same_prefix_pair(use_fp16=True)

    path_prefix = args.qid_smtid_docids_path.split(".")[0]
    out_path = path_prefix + f"_teacher_score_{args.local_rank}.train.json"
    with open(out_path, "w") as fout:
        ujson.dump(qid_to_smtid_to_rankdata, fout)
    
def cross_encoder_rerank_for_qid_smtid_docids_2(args):
    if os.path.exists(os.path.join(args.out_dir, "qid_smtid_docids_teacher_score.train.json")):
        print("old qid_smtid_docids_teacher_score.train.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_smtid_docids_teacher_score.train.json"))

    qid_to_smtid_to_rerank = {}
    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "qid_smtid_docids_teacher_score" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count()
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rerank = ujson.load(fin)
        
        for qid in sub_qid_to_rerank:
            for smtid in sub_qid_to_rerank[qid]:
                if qid not in qid_to_smtid_to_rerank:
                    qid_to_smtid_to_rerank[qid] = {smtid: sub_qid_to_rerank[qid][smtid]}
                else:
                    if smtid not in qid_to_smtid_to_rerank[qid]:
                        qid_to_smtid_to_rerank[qid][smtid] = sub_qid_to_rerank[qid][smtid]
                    else:
                        qid_to_smtid_to_rerank[qid][smtid] += sub_qid_to_rerank[qid][smtid]

    print("total size of qids = {}".format(len(qid_to_smtid_to_rerank)))
    print("average smtids per qid = {:.3f}".format(np.mean([len(xs) for xs in qid_to_smtid_to_rerank.values()])))
    with open(os.path.join(args.out_dir, "qid_smtid_docids_teacher_score.train.json"), "w") as fout:
        ujson.dump(qid_to_smtid_to_rerank, fout)

    for sub_path in sub_rerank_paths:
        sub_path = os.path.join(args.out_dir, sub_path)
        os.remove(sub_path)

if __name__ == "__main__":
    parser = HfArgumentParser((RerankArguments))
    args = parser.parse_args_into_dataclasses()[0]

    if args.task == "rerank_for_eval":
        rerank_for_eval(args)
    elif args.task == "rerank_for_create_trainset":
        rerank_for_create_trainset(args)
    elif args.task == "rerank_for_create_trainset_2":
        rerank_for_create_trainset_2(args)
    elif args.task == "rerank_for_evaluate_2":
        rerank_for_evaluate_2(args)
    elif args.task == "assign_scores_for_pseudo_queries":
        assign_scores_for_pseudo_queries(args)
    elif args.task == "assign_scores_for_pseudo_queries_2":
        assign_scores_for_pseudo_queries_2(args)
    elif args.task == "query_to_docid_rerank_for_qid_smtids":
        query_to_docid_rerank_for_qid_smtids(args)
    elif args.task == "query_to_docid_rerank_for_qid_smtids_2":
        query_to_docid_rerank_for_qid_smtids_2(args)
    elif args.task == "teacher_rerank_for_qid_smtids":
        teacher_rerank_for_qid_smtids(args)
    elif args.task == "teacher_rerank_for_qid_smtids_2":
        teacher_rerank_for_qid_smtids_2(args)
    elif args.task == "cross_encoder_rerank_for_same_prefix_docid":
        cross_encoder_rerank_for_same_prefix_docid(args)
    elif args.task == "cross_encoder_rerank_for_same_prefix_docid_2":
        cross_encoder_rerank_for_same_prefix_docid_2(args)
    elif args.task == "cross_encoder_rerank_for_same_reldocid_hard_docids":
        cross_encoder_rerank_for_same_reldocid_hard_docids(args)
    elif args.task == "cross_encoder_rerank_for_same_reldocid_hard_docids_2":
        cross_encoder_rerank_for_same_reldocid_hard_docids_2(args)
    elif args.task == "cross_encoder_rerank_for_qid_smtid_docids":
        cross_encoder_rerank_for_qid_smtid_docids(args)
    elif args.task == "cross_encoder_rerank_for_qid_smtid_docids_2":
        cross_encoder_rerank_for_qid_smtid_docids_2(args)
    else:
        raise NotImplementedError