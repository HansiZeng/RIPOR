import os 
import ujson 
import os 
import pandas as pd 
import random 
from tqdm import tqdm
random.seed(4680)

def from_bm25_run_to_rundata(bm25_run_path):
    names = ["qid", "Q0", "docid", "rank", "score", "model_name"]

    bm25_data = pd.read_csv(bm25_run_path, names=names, delimiter=" ")

    print(bm25_data.head())

    qids = bm25_data.qid
    docids = bm25_data.docid 
    scores = bm25_data.score 

    assert len(qids) == len(docids) == len(scores)

    qid_to_rankdata = {}
    for qid, docid, score in zip(qids, docids, scores):
        qid = str(qid)
        docid = str(docid)
        score = float(score)

        if qid not in qid_to_rankdata:
            qid_to_rankdata[qid] = {docid: score}
        else:
            qid_to_rankdata[qid][docid] = score 

    return qid_to_rankdata


if __name__ == "__main__":
    train_qrel_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/qrels.json" 
    run_path = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/run.json"
    out_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/ce_train/"
    neg_sample = 50

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if "bm25_run" in run_path and run_path.endswith(".run"):
        print("run path is bm25_run")
        run_data = from_bm25_run_to_rundata(run_path)
    else:
        print("run path is rankdata_run")
        with open(run_path) as fin:
            run_data = ujson.load(fin)

    with open(train_qrel_path) as fin:
        train_qrel = ujson.load(fin)

    train_examples = []
    for qid in tqdm(train_qrel, total=len(train_qrel)):
        for rel_docid in train_qrel[qid]:
            neg_docids = random.sample(list(run_data[qid].keys()), k=neg_sample)
            
            for neg_docid in neg_docids:
                train_examples.append((qid, rel_docid, 1))
                train_examples.append((qid, neg_docid, 0))
    
    print("size of train_examples = {}".format(len(train_examples)))
    print("size of qids in train_examples = {}".format(len(train_examples)//(neg_sample*2)))
    print("neg samples per query = {}".format(len(run_data[qid])))
    random.shuffle(train_examples)
    with open(os.path.join(out_dir, f"bce_examples.{neg_sample}_{neg_sample}.train.tsv"), "w") as fout:
        for qid, docid, label in train_examples:
            fout.write(f"{qid}\t{docid}\t{label}\n")