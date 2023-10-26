import ujson 
import os 
import numpy as np 
from tqdm import tqdm

def read_teacher_path(path):
    qid_to_rankdata = {}
    with open(path) as fin:
        for line in fin:
            example = ujson.loads(line)
            qid = example["qid"]
            docids = example["docids"]
            scores = example["scores"]

            #assert qid not in qid_to_rankdata
            qid_to_rankdata[qid] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[qid][docid] = score
    
    print("size of qid_to_rankdata = {}, average docids per query = {:.3f}".format(
        len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()]), 
    ))
    print("lengths' distribution: ", np.quantile([len(xs) for xs in qid_to_rankdata.values()], [0.0, 0.1, .25, .5, .75, .9, 1.0]))

    return qid_to_rankdata

teacher_score_path_1 = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1/out/MSMARCO_TRAIN/qid_docids_teacher_scores.train.json"
teacher_score_path_2 = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json"
out_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1/out/MSMARCO_TRAIN/"

qid_to_rankdata_1 = read_teacher_path(teacher_score_path_1)
qid_to_rankdata_2 = read_teacher_path(teacher_score_path_2)

#assert len(qid_to_rankdata_2)  len(qid_to_rankdata_1), (len(qid_to_rankdata_2), len(qid_to_rankdata_1))

new_qid_to_rankdata = {}
for qid, rankdata in tqdm(qid_to_rankdata_1.items(), total=len(qid_to_rankdata_1)):
    new_qid_to_rankdata[qid] = rankdata

    for docid, score in  qid_to_rankdata_2[qid].items():
        if docid not in new_qid_to_rankdata[qid]:
            new_qid_to_rankdata[qid][docid] = score 
        if len(new_qid_to_rankdata[qid]) >= 60:
            break 

ignored_num = 0
for qid in qid_to_rankdata_2:
    if qid not in new_qid_to_rankdata:
        rankdata = qid_to_rankdata_2[qid]
        sorted_rankdata = sorted(rankdata.items(), key=lambda x: x[1], reverse=True)
        new_qid_to_rankdata[qid] = {}
        
        for docid, score in sorted_rankdata:
            new_qid_to_rankdata[qid][docid] = score 
            if len(new_qid_to_rankdata[qid]) == 60:
                break
        ignored_num += 1
print("ignored_num: ", ignored_num)

train_examples = []
lengths = []
for qid, rankdata in tqdm(new_qid_to_rankdata.items(), total=len(new_qid_to_rankdata)):
    sorted_rankdata = sorted(rankdata.items(), key=lambda x: x[1], reverse=True)
    docids = []
    scores = []
    for docid, score in sorted_rankdata:
        docids.append(docid)
        scores.append(score)
    lengths.append(len(docids))

    assert scores[2] >= scores[3] and scores[10] >= scores[11], (scores[2], scores[3], scores[10], scores[11])
    train_examples.append({
        "qid": qid,
        "docids": docids,
        "scores": scores
    })

print("distribution of train examples' length: ", np.quantile(lengths, [0.0, .1, .25, .5, .75, .9, 1.0]))
with open(os.path.join(out_dir, "teacher_rerank_scores_merge_from_dense_hard.train.json"), "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")


