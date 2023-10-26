import ujson 
import os 
from tqdm import tqdm 
import numpy as np

decay_to_factor = {
    1: {4: 4./32, 8: 4./32 + 8./32, 16: 4./32 + 8./32 + 16./32},
    2: {4: 1./2, 8: 1./2 + (1./2)**2, 16: 1./2 + (1./2)**2 + (1./2)**3, 32: 1.},
    8: {4: 0.8, 8: 0.8 + (1-0.8)*0.8, 16: 0.8 + (1-0.8)*0.8 + (1-0.8-0.2*0.8)*0.8, 32: 1.}
}

decay = 2
max_new_token = 32
factor = decay_to_factor[decay][max_new_token]
print("max_new_token: ", max_new_token, "factor: ", factor) 

root_dir = f"/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_{max_new_token}_dcy_2"
docid_to_smtid_path = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json"

qid_smtid_rerank_path = os.path.join(root_dir, f"sub_smtid_{max_new_token}_out/qid_smtid_docids_teacher_score.train.json")
qid_to_reldocid_to_score_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-1M/doc2query/all_train_queries/qid_to_reldocid_to_score.json"

out_dir = os.path.join(root_dir, "sub_smtid_train_decay2")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(docid_to_smtid_path) as fin:
    docid_to_smtids = ujson.load(fin)

with open(qid_to_reldocid_to_score_path) as fin:
    qid_to_reldocid_to_score = ujson.load(fin)

qid_to_relsmtid_to_score = {}
for qid in qid_to_reldocid_to_score:
    qid_to_relsmtid_to_score[qid] = {}
    for reldocid, score in qid_to_reldocid_to_score[qid].items():
        rel_smtids = docid_to_smtids[reldocid]
        assert rel_smtids[0] == -1, len(rel_smtids) == 33 
        relsmtid = "_".join([str(x) for x in rel_smtids[1:1+max_new_token]])
        qid_to_relsmtid_to_score[qid][relsmtid] = score 

print("size of qid in qid_to_relsmtid_to_score and its reldocid version = {}, {}".format(len(qid_to_relsmtid_to_score), 
                                                            len(qid_to_reldocid_to_score)))

with open(qid_smtid_rerank_path) as fin:
    qid_to_smtid_to_rerank = ujson.load(fin)

qid_to_smtid_to_score = {}
for qid in qid_to_smtid_to_rerank:
    qid_to_smtid_to_score[qid] = {}
    for smtid, rerank in qid_to_smtid_to_rerank[qid].items():
        score = max([xs[1] for xs in rerank]) * factor 
        qid_to_smtid_to_score[qid][smtid] = score 

print("size of qid_to_smtid_to_score = {}".format(len(qid_to_smtid_to_score)))
# sanity check 
assert len(smtid.split("_")) == max_new_token, (len(smtid.split("_")))

train_examples = []
rel_scores = []
rel_rates = []
largest_scores = []
ignore_qids = 0
smtid_lengths = []
for qid, relsmtid_to_score in tqdm(qid_to_relsmtid_to_score.items(), total=len(qid_to_relsmtid_to_score)):
    if qid not in qid_to_smtid_to_score:
        ignore_qids += 1
        continue 

    sorted_pairs = sorted(qid_to_smtid_to_score[qid].items(), key=lambda x: x[1], reverse=True)
    smtids, scores = [], []
    for smtid, score in sorted_pairs:
        smtids.append(smtid)
        scores.append(score)
    for relsmtid, score in relsmtid_to_score.items():
        if relsmtid not in smtids:
            train_examples.append({"qid": qid,
                                   "smtids": [relsmtid] + smtids,
                                   "scores": [score] + scores})
            rel_rates.append(0)
        else:
            train_examples.append({"qid": qid,
                                   "smtids": smtids,
                                   "scores": scores})
            rel_rates.append(1)

        rel_scores.append(score)
        largest_scores.append(max(scores))
        smtid_lengths.append(len(smtids))

quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
print("size of train_examples = {}".format(len(train_examples)))
print("distribution of relevant scores: ", np.quantile(rel_scores, quantiles))
print("distribution of largest_scores when relevant scores not consider: ", np.quantile(largest_scores, quantiles))
print("total qids in rel qid = {}, ignore_qids = {}, ratio = {:.3f}".format(len(qid_to_relsmtid_to_score), ignore_qids,
                                                                            ignore_qids / len(qid_to_relsmtid_to_score)))
print("distribution of length of smtids per query: ", np.quantile(smtid_lengths, quantiles))
print("relrates: {:.3f}".format(np.sum(rel_rates) / len(rel_rates)))

with open(os.path.join(out_dir, f"qid_smtids_scores_{max_new_token}.train.json"), "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")


                            