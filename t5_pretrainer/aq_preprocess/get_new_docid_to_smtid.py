import ujson 
import os 
import numpy as np

in_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/reorg_assign/"

with open(os.path.join(in_dir, "docid_to_prefix.json")) as fin:
    docid_to_prefix = ujson.load(fin)

with open(os.path.join(in_dir, "docid_to_pfids.json")) as fin:
    docid_to_pfids = ujson.load(fin)

# we change the smtid of docid in nr_qids 
modified_docid_to_smtid = {}
for qid, smtids in nr_qid_to_smtids.items():
    for smtid in smtids:
        docids = smtid_to_docids[smtid]

        l_smtid = smtid.split("_")
        f_smtid = nr_qid_to_first_smtid[qid].split("_")
        assert len(l_smtid) == len(f_smtid) == 32 
        new_smtid = "_".join([str(x) for x in f_smtid[:8]]) +  "_" + "_".join([str(x) for x in l_smtid[8:]])

        for docid in docids:
            modified_docid_to_smtid[docid] = new_smtid

print("size of docid is modified = {}, ratio = {:.3f}".format(len(modified_docid_to_smtid), len(modified_docid_to_smtid) / len(docid_to_smtid)))

new_docid_to_smtid = {}
for docid in docid_to_smtid:
    if docid in modified_docid_to_smtid:
        new_docid_to_smtid[docid] = modified_docid_to_smtid[docid]
    else:
        new_docid_to_smtid[docid] = docid_to_smtid[docid]

assert len(new_docid_to_smtid) == len(docid_to_smtid)

old_smtid_to_new_smtid = {}
for docid in docid_to_smtid:
    old_smtid = docid_to_smtid[docid]
    new_smtid = new_docid_to_smtid[docid]

    old_smtid_to_new_smtid[old_smtid] = new_smtid

new_smtid_to_docids = {}
for docid, smtid in new_docid_to_smtid.items():
    if smtid not in new_smtid_to_docids:
        new_smtid_to_docids[smtid] = [docid]
    else:
        new_smtid_to_docids[smtid] += [docid]

docid_lengths = [len(x) for x in new_smtid_to_docids.values()]
print("distribution of docids per smtid: ", np.quantile(docid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))


out_docid_to_smtid = {}
for docid, smtid in new_docid_to_smtid.items():
    l_smtid = [-1] + [int(x) for x in smtid.split("_")]
    out_docid_to_smtid[docid] = l_smtid
    assert len(l_smtid) == 33, len(l_smtid)



print("size of out_docid_to_smtid = {}".format(len(out_docid_to_smtid)))
with open(os.path.join(out_dir, "docid_to_smtid.json"), "w") as fout:
    ujson.dump(out_docid_to_smtid, fout)

with open(os.path.join(out_dir, "old_smtid_to_new_smtid.json"), "w") as fout:
    ujson.dump(old_smtid_to_new_smtid, fout)