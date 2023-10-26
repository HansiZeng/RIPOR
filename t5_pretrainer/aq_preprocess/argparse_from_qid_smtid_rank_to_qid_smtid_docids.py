import ujson
import os 
from tqdm import tqdm 
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=None, type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    root_dir = args.root_dir
    qid_smtid_rank_path = os.path.join(root_dir, "qid_smtid_rankdata.json")
    qid_smtid_docids_path = os.path.join(root_dir, "qid_smtid_docids.train.json")
    print("root_dir: ", root_dir)

    with open(qid_smtid_rank_path) as fin:
        qid_to_smtid_to_rank = ujson.load(fin)

    ignore_num = 0
    total_num = 0
    qid_to_smtid_to_docids = {}
    for qid in tqdm(qid_to_smtid_to_rank, total=len(qid_to_smtid_to_docids)):
        qid_to_smtid_to_docids[qid] = {}
        for smtid in qid_to_smtid_to_rank[qid]:
            rank_data = qid_to_smtid_to_rank[qid][smtid]
            if len(rank_data) == 0:
                ignore_num += 1
            else:
                qid_to_smtid_to_docids[qid][smtid] = list(rank_data.keys())
            total_num += 1

    print("total_num = {}, ignore_num = {}, ratio = {:.3f}".format(total_num, ignore_num, ignore_num / total_num))
    print("distribution of smtids per query: ", np.quantile([len(xs) for xs in qid_to_smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    with open(qid_smtid_docids_path, "w") as fout:
        ujson.dump(qid_to_smtid_to_docids, fout)