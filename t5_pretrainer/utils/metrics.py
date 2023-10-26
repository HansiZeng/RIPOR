from collections import Counter
import json

import pytrec_eval
from pytrec_eval import RelevanceEvaluator
from .utils import from_qrel_to_qsmtid_rel


def truncate_run(run, k):
    """truncates run file to only contain top-k results for each query"""
    temp_d = {}
    for q_id in run:
        sorted_run = {k: v for k, v in sorted(run[q_id].items(), key=lambda item: item[1], reverse=True)}
        temp_d[q_id] = {k: sorted_run[k] for k in list(sorted_run.keys())[:k]}
    return temp_d


def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    
    mrr = evaluator.evaluate(truncated)
    if agg:
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr

def recall_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recall"})
    out_eval = evaluator.evaluate(run)

    total_v = 0.
    included_k = f"recall_{k}"
    for q, v_dict in out_eval.items():
        for k, v in v_dict.items():
            if k == included_k:
                total_v += v 

    return total_v / len(out_eval)            

                


def evaluate(run, qrel, metric, agg=True, select=None):
    assert metric in pytrec_eval.supported_measures, print("provide valid pytrec_eval metric")
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    res = Counter({})
    if agg:
        for d in out_eval.values():  # when there are several results provided (e.g. several cut values)
            res += Counter(d)
        res = {k: v / len(out_eval) for k, v in res.items()}
        if select is not None:
            string_dict = "{}_{}".format(metric, select)
            if string_dict in res:
                return res[string_dict]
            else:  # If the metric is not on the dict, say that it was 0
                return 0
        else:
            return res
    else:
        return out_eval
    
def load_and_evaluate(qrel_file_path, run_file_path, metric):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)
    # for trec, qrel_binary.json should be used for recall etc., qrel.json for NDCG.
    # if qrel.json is used for binary metrics, the binary 'splits' are not correct
    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    else:
        res = evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res
    
def load_and_evaluate_for_qid_smtid(qrel_file_path, run_file_path, metric, docid_to_smtid_path, truncate_smtid=False):
    qrel = from_qrel_to_qsmtid_rel(docid_to_smtid_path, qrel_file_path, truncate_smtid=truncate_smtid)
    with open(run_file_path) as reader:
        run = json.load(reader)

    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10)
        print("MRR@10:", res)
        return {"mrr_10": res}
    else:
        res = evaluate(run, qrel, metric=metric)
        print(metric, "==>", res)
        return res


def init_eval(metric):
    if metric not in ["MRR@10", "recall@10", "recall@50", "recall@100", "recall@200", "recall@500", "recall@1000",]:
        raise NotImplementedError("provide valid metric")
    if metric == "MRR@10":
        return lambda x, y: mrr_k(x, y, k=10, agg=True)
    else:
        return lambda x, y: evaluate(x, y, metric="recall", agg=True, select=metric.split('@')[1])
    

def get_fsmtid_hit_rate(batch_qids, prefx_smtids, qid_to_reldata, docid_to_smtids):
    qid_to_hitrate = {}
    for qid, psmtid in zip(batch_qids, prefx_smtids):
        assert len(psmtid) == 1 
        fsmtid = int(psmtid[0])
        

        rel_docids = [x for x in qid_to_reldata[qid]]
        rel_fsmtids = set()
        for docid in rel_docids:
            assert docid_to_smtids[docid][0] == -1, docid_to_smtids[docid]
            rel_fsmtids.add(int(docid_to_smtids[docid][1]))
        
        if fsmtid in rel_fsmtids:
            qid_to_hitrate[qid] = 1. 
        else:
            qid_to_hitrate[qid] = 0. 
    
    return qid_to_hitrate
