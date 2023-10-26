import os 
import ujson 
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docid_to_smtid_path", default=None, type=str)
    parser.add_argument("--teacher_score_path", default=None, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--max_new_token", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    docid_to_smtid_path = args.docid_to_smtid_path
    teacher_score_path = args.teacher_score_path
    out_dir = args.out_dir 
    max_new_token = args.max_new_token

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    assert max_new_token in {4, 8, 16, 32}

    with open(docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    docid_to_smtid = {}
    for docid, smtids in docid_to_smtids.items():
        assert len(smtids) in [5, 9, 17,33], len(smtids)
        smtid = "_".join([str(x) for x in smtids[1:1+max_new_token]])
        docid_to_smtid[docid] = smtid
    print("size of docid_to_smtid = {}".format(len(docid_to_smtid)))

    qid_to_smtid_to_docids = {}
    with open(teacher_score_path) as fin:
        for line in fin:
            example = ujson.loads(line)
            qid = example["qid"]
            docids = example["docids"]

            if qid not in qid_to_smtid_to_docids:
                qid_to_smtid_to_docids[qid] = {}
            
            for docid in docids:
                smtid = docid_to_smtid[docid]
                if smtid not in qid_to_smtid_to_docids[qid]:
                    qid_to_smtid_to_docids[qid][smtid] = [docid]
                else:
                    qid_to_smtid_to_docids[qid][smtid] += [docid]

    docid_lengths = []
    for qid in qid_to_smtid_to_docids:
        for smtid in qid_to_smtid_to_docids[qid]:
            docid_lengths.append(len(qid_to_smtid_to_docids[qid][smtid]))
    print("distribution of docid_lengths: ", np.quantile(docid_lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(out_dir, f"qid_smtid_docids.train.json"), "w") as fout:
        ujson.dump(qid_to_smtid_to_docids, fout)