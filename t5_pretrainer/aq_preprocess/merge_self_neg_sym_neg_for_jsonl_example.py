import ujson 
import os 
from tqdm import tqdm
import numpy as np 

max_new_token = 32
decay = 2

root_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/"
docid_to_smtid_path = os.path.join(root_dir, "t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json")
qid_smtid_rank_path = os.path.join(root_dir, "t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/sub_smtid_32_out/qid_smtid_docids_teacher_score.train.json")
qid_reldocid_rank_path = os.path.join(root_dir, "t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/symmetric_hard_neg/qid_to_reldocid_to_topids_teacher_score.json")
qid_to_reldocid_to_score_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-1M/doc2query/all_train_queries/qid_to_reldocid_to_score.json"
out_dir = os.path.join(root_dir, f"t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/syn_sfn_sub_smtids_train_decay{decay}")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

treshold = 80
decay_to_factor = {
    1: {4: 4./32, 8: 4./32 + 8./32, 16: 4./32 + 8./32 + 16./32},
    2: {4: 1./2, 8: 1./2 + (1./2)**2, 16: 1./2 + (1./2)**2 + (1./2)**3, 32: 1.},
    8: {4: 0.8, 8: 0.8 + (1-0.8)*0.8, 16: 0.8 + (1-0.8)*0.8 + (1-0.8-0.2*0.8)*0.8, 32: 1.}
}
factor = decay_to_factor[decay][max_new_token]
print("factor: ", factor, "max_new_token: ", max_new_token)

docid_to_smtid = {}
with open(docid_to_smtid_path) as fin:
    docid_to_smtids = ujson.load(fin)
for docid, smtids in docid_to_smtids.items():
    assert len(smtids) == 33 
    smtid = "_".join([str(x) for x in smtids[1:1+max_new_token]])
    docid_to_smtid[docid] = smtid 

with open(qid_smtid_rank_path) as fin:
    qid_to_smtid_to_rank = ujson.load(fin)

with open(qid_reldocid_rank_path) as fin:
    qid_to_reldocid_to_rank = ujson.load(fin)

with open(qid_to_reldocid_to_score_path) as fin:
    qid_to_reldocid_to_score = ujson.load(fin)

qid_to_relsmtid_to_smtid_to_score = {}
qid_to_relsmtid_to_score = {}
lengths_1 = []
lengths_2 = []
lengths_all = []
for qid in tqdm(qid_to_reldocid_to_rank, total=len(qid_to_reldocid_to_rank)):
    qid_to_relsmtid_to_smtid_to_score[qid] = {}
    qid_to_relsmtid_to_score[qid] = {}
    for reldocid in qid_to_reldocid_to_rank[qid]:
        relsmtid = docid_to_smtid[reldocid]
        qid_to_relsmtid_to_smtid_to_score[qid][relsmtid] = {}
        qid_to_relsmtid_to_score[qid] = {relsmtid: qid_to_reldocid_to_score[qid][reldocid]}
        for (docid, score) in qid_to_reldocid_to_rank[qid][reldocid]:
            smtid = docid_to_smtid[docid]
            
            if smtid not in qid_to_relsmtid_to_smtid_to_score[qid][relsmtid]:
                qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid] = factor * score 
            else:
                qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid] = max(factor*score, qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid])
        lengths_1.append(len(qid_to_reldocid_to_rank[qid][reldocid]))
        for smtid in qid_to_smtid_to_rank[qid]:
            score = max([x[1] for x in qid_to_smtid_to_rank[qid][smtid]])
            if smtid not in qid_to_relsmtid_to_smtid_to_score[qid][relsmtid]:
                qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid] = factor * score 
            else:
                qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid] = max(factor*score, qid_to_relsmtid_to_smtid_to_score[qid][relsmtid][smtid])
        lengths_2.append(len(qid_to_smtid_to_rank[qid]))
        lengths_all.append(len(qid_to_relsmtid_to_smtid_to_score[qid][relsmtid]))

train_examples = []
length_example = []
for qid in qid_to_relsmtid_to_score:
    for relsmtid, relscore in qid_to_relsmtid_to_score[qid].items():
        example = {"qid": qid, "smtids": [relsmtid], "scores": [relscore]}
        for smtid, score in sorted(qid_to_relsmtid_to_smtid_to_score[qid][relsmtid].items(), key=lambda x: x[1], reverse=True):
            if smtid != relsmtid:
                example["smtids"] += [smtid]
                example["scores"] += [score]

            if len(example["smtids"]) == treshold:
                break 
        train_examples.append(example)
        length_example.append(len(example["smtids"]))

quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
print("distribution of lengths_1: ", np.quantile(lengths_1, quantiles))
print("distribution of lengths_2: ", np.quantile(lengths_2, quantiles))
print("distribution of lengths_all: ", np.quantile(lengths_all, quantiles))
print("avg lengths of them = {:.3f}, {:.3f}, {:.3f}".format(np.mean(lengths_1), np.mean(lengths_2), np.mean(lengths_all)))
print("distribution of length_example: ", np.quantile(length_example, quantiles))

with open(os.path.join(out_dir, f"syn_sfn_qid_smtid_scores_{max_new_token}.train.json"), "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")