import os 
import ujson 
from tqdm import tqdm 
import numpy as np

qid_to_reldocid_to_score_path = "./data/msmarco-full/all_train_queries/qid_to_reldocid_to_score.json"
qid_pids_rerank_scores_path = "./experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/qid_docids_teacher_scores.train.json"

out_path = "./experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json"
with open(qid_to_reldocid_to_score_path) as fin:
    qid_to_reldocid_to_score = ujson.load(fin)

train_examples = []
top_scores = []
ignore_num = 0
with open(qid_pids_rerank_scores_path) as fin:
    for line in tqdm(fin):
        example = ujson.loads(line)
        qid = example["qid"]
        docids = example["docids"]
        scores = example["scores"]

        reldocid_to_score = qid_to_reldocid_to_score[qid]
        for reldocid, score in reldocid_to_score.items():
            if reldocid not in docids:
                train_examples.append({
                    "qid": qid,
                    "docids": [reldocid] + docids,
                    "scores": [score] + scores
                })
                ignore_num += 1
            else:
                train_examples.append({
                    "qid": qid,
                    "docids": docids,
                    "scores": scores
                })

            top_scores.append(train_examples[-1]["scores"][0])

print("size of train_examples = {}".format(len(train_examples)))
print("distribution of top scores: ", np.quantile(top_scores, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]))
print("ignored num = {}".format(ignore_num))
with open(out_path, "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")
