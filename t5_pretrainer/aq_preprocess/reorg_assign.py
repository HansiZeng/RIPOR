import os 
import ujson 
from tqdm.auto import tqdm 
import numpy as np 
from scipy.sparse import coo_matrix

docid_to_smtid_path = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5_docid_gen_encoder_1/aq_smtid/docid_to_smtid.json"
qid_smtid_rerank_path = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/sub_smtid_32_out/qid_smtid_docids_teacher_score.train.json"
qid_to_reldocid_to_score_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-1M/doc2query/all_train_queries/qid_to_reldocid_to_score.json"
qid_smtid_rank_path = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/sub_smtid_16_out/qid_smtid_rankdata.json"

with open(docid_to_smtid_path) as fin:
    docid_to_smtids = ujson.load(fin)

with open(qid_to_reldocid_to_score_path) as fin:
    qid_to_reldocid_to_score = ujson.load(fin)

with open(qid_smtid_rerank_path) as fin:
    qid_to_smtid_to_rerank = ujson.load(fin)

with open(qid_smtid_rank_path) as fin:
    qid_to_smtid_to_rank = ujson.load(fin)


topk=5
prefix_to_smtid_to_docids = {}
prefix_to_docids = {}
smtid_to_docids = {}
for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
    assert len(smtids) == 33 
    prefix = "_".join([str(x) for x in smtids[1:3]])
    smtid = "_".join([str(x) for x in smtids[1:]])

    if prefix not in prefix_to_smtid_to_docids:
        prefix_to_smtid_to_docids[prefix] = {smtid: [docid]}
    else:
        if smtid not in prefix_to_smtid_to_docids[prefix]:
            prefix_to_smtid_to_docids[prefix][smtid] = [docid]
        else:
            prefix_to_smtid_to_docids[prefix][smtid] += [docid]
    
    if prefix not in prefix_to_docids:
        prefix_to_docids[prefix] = [docid]
    else:
        prefix_to_docids[prefix] += [docid]

    if smtid not in smtid_to_docids:
        smtid_to_docids[smtid] = [docid]
    else:
        smtid_to_docids[smtid] += [docid]

print("size of prefix = {}".format(len(prefix_to_docids)))
assert len(prefix_to_smtid_to_docids) == len(prefix_to_docids)
print("size of smtid = {}".format(len(smtid_to_docids)))
print("distribution of leaf sizes: ", np.quantile([len(xs) for xs in prefix_to_smtid_to_docids.values()], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
print("mean of leaf size = {:.3f}".format(np.mean([len(xs) for xs in prefix_to_smtid_to_docids.values()])))


pos_scores = []
qid_to_pos_docids = {}
for qid, smtid_to_rerank in tqdm(qid_to_smtid_to_rerank.items(), total=len(qid_to_smtid_to_rerank)):
    qid_to_pos_docids[qid] = []
    docid_score_pairs = []
    for rerank in smtid_to_rerank.values():
        for docid, score in rerank:
            docid_score_pairs.append((docid, score))

    for docid, score in sorted(docid_score_pairs, key=lambda x: x[1], reverse=True)[:topk]:
        if score > 0.5:
            qid_to_pos_docids[qid].append(docid)
        pos_scores.append(score)
    
for qid in qid_to_reldocid_to_score:
    qid_to_pos_docids[qid] = []
    for reldocid in qid_to_reldocid_to_score[qid]:
        if reldocid not in qid_to_pos_docids[qid]:
            qid_to_pos_docids[qid].append(reldocid)

print("distribution of pos_scores: ", np.quantile(pos_scores, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

qid_to_prefix_to_score = {}
qid_to_top_prefixs = {}
for qid, smtid_to_rank in tqdm(qid_to_smtid_to_rank.items(), total=len(qid_to_smtid_to_rank)):
    qid_to_prefix_to_score[qid] = {}
    for smtid, ranks in smtid_to_rank.items():
        prefix = "_".join([str(x) for x in smtid.split("_")[:2]])
        assert prefix in prefix_to_smtid_to_docids, prefix  
        score = max(ranks.values())

        if prefix not in qid_to_prefix_to_score[qid]:
            qid_to_prefix_to_score[qid][prefix] = score
        else:
            qid_to_prefix_to_score[qid][prefix] = max(score, qid_to_prefix_to_score[qid][prefix])

    sorted_pairs = sorted(qid_to_prefix_to_score[qid].items(), key=lambda x: x[1], reverse=True)
    qid_to_top_prefixs[qid] = [x[0] for x in sorted_pairs[:topk]]

print("size of qid_to_top_prefixs = {}".format(len(qid_to_top_prefixs)))


docid_to_docidx = {}
docidx_to_docid = {}
for docid in docid_to_smtids:
    docidx = len(docid_to_docidx)
    docid_to_docidx[docid] = docidx
    docidx_to_docid[docidx] = docid

print("size of docid_to_docidx = {}".format(len(docid_to_docidx)))
assert len(docid_to_docidx) == len(docidx_to_docid)

qid_to_qidx = {}
qidx_to_qid = {}
for qid in qid_to_pos_docids:
    qidx = len(qid_to_qidx)
    qid_to_qidx[qid] = qidx 
    qidx_to_qid[qidx] = qid 
print("size of qid_to_qidx = {}".format(len(qid_to_qidx)))
assert len(qidx_to_qid) == len(qid_to_qidx)

prefix_to_pfidx = {}
pfidx_to_prefix = {}
for prefix in prefix_to_docids:
    pfidx = len(prefix_to_pfidx)
    prefix_to_pfidx[prefix] = pfidx 
    pfidx_to_prefix[pfidx] = prefix 

print("size of prefix_to_pfidx = {}".format(len(prefix_to_pfidx)))
assert len(prefix_to_pfidx) == len(pfidx_to_prefix)


Q, N, L = len(qid_to_qidx), len(docid_to_docidx), len(prefix_to_pfidx)


Y_rows, Y_cols, Y_data = [], [], []
for qid, pos_docids in tqdm(qid_to_pos_docids.items(), total=len(qid_to_pos_docids)):
    qidx = qid_to_qidx[qid] 
    for docid in pos_docids:
        docidx = docid_to_docidx[docid]

        Y_rows.append(qidx)
        Y_cols.append(docidx)
        Y_data.append(1.)

M_rows, M_cols, M_data = [], [], []
for qid, top_prefixes in tqdm(qid_to_top_prefixs.items(), total=len(qid_to_top_prefixs)):
    qidx = qid_to_qidx[qid]
    for prefix in top_prefixes:
       pfidx = prefix_to_pfidx[prefix]

       M_rows.append(qidx)
       M_cols.append(pfidx)
       M_data.append(1.)

Y_sparse = coo_matrix((Y_data, (Y_rows, Y_cols)), shape=(Q, N))
M_sparse = coo_matrix((M_data, (M_rows, M_cols)), shape=(Q, L))
C = Y_sparse.transpose().dot(M_sparse)

docid_to_prefix = {}
for prefix, docids in prefix_to_docids.items():
    for docid in docids:
        docid_to_prefix[docid] = prefix 
print("size of docid_to_prefix = {}".format(len(docid_to_prefix)))

num_rows = C.shape[0]
docid_to_pfids = {}
for i in tqdm(range(num_rows), total=num_rows):
    row = C[i].toarray().ravel()
    top_5_indices = row.argsort()[-5:][::-1]  # Take top 2 and reverse to descending order
    
    docid = docidx_to_docid[i]
    pfids = [pfidx_to_prefix[idx] for idx in top_5_indices]
    docid_to_pfids[docid] = pfids

out_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/reorg_assign/"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(os.path.join(out_dir, "docid_to_prefix.json"), "w") as fout:
    ujson.dump(docid_to_prefix, fout)

with open(os.path.join(out_dir, "docid_to_pfids.json"), "w") as fout:
    ujson.dump(docid_to_pfids, fout)
