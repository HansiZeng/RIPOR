import os 
import ujson 
from tqdm import tqdm
import numpy as np 
import argparse

src_new_token = 16
tgt_new_token = 32

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--src_qid_smtid_rankdata_path", default="./experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1/sub_smtid_16_out/qid_smtid_rankdata.json")
    parser.add_argument("--out_dir", default="./experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1/sub_smtid_32_out/")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_dir = args.data_dir
    print("source new token = {}, target_new_token = {}".format(src_new_token, tgt_new_token))

    docid_to_smtid_path = os.path.join(data_dir, "aq_smtid/docid_to_smtid.json")
    src_qid_smtid_rankdata_path = args.src_qid_smtid_rankdata_path
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    with open(src_qid_smtid_rankdata_path) as fin:
        src_qid_smtid_rankdata = ujson.load(fin)

    tgt_smtid_to_docids = {}
    src_smtid_to_tgt_smtids = {}
    for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
        assert smtids[0] == -1 and len(smtids) == 33, (smtids)

        src_smtid = "_".join([str(x) for x in smtids[1:1+src_new_token]])
        tgt_smtid = "_".join([str(x) for x in smtids[1:1+tgt_new_token]])
        
        if tgt_smtid not in tgt_smtid_to_docids:
            tgt_smtid_to_docids[tgt_smtid] = [docid]
        else:
            tgt_smtid_to_docids[tgt_smtid] += [docid]
        
        if src_smtid not in src_smtid_to_tgt_smtids:
            src_smtid_to_tgt_smtids[src_smtid] = [tgt_smtid]
        else:
            if tgt_smtid in src_smtid_to_tgt_smtids[src_smtid]:
                continue
            src_smtid_to_tgt_smtids[src_smtid] += [tgt_smtid]

    ignore_num = 0
    total_num = 0
    tgt_smtid_total_num = 0
    docids_total_num = 0
    tgt_qid_smtid_docids = {}
    for qid in tqdm(src_qid_smtid_rankdata, total=len(src_qid_smtid_rankdata)):
        tgt_qid_smtid_docids[qid] = {}
        for src_smtid in src_qid_smtid_rankdata[qid]:
            if src_smtid not in src_smtid_to_tgt_smtids:
                ignore_num += 1
                continue
            total_num += 1 

            tgt_smtids = src_smtid_to_tgt_smtids[src_smtid]
            for tgt_smtid in tgt_smtids:
                #if tgt_smtid == '136_121_208_223_204_250_119_211_242_129_239_43_39_11_210_168':
                    #print(src_smtid, src_smtid_to_tgt_smtids[src_smtid])
                assert tgt_smtid not in tgt_qid_smtid_docids[qid], (tgt_smtid, tgt_qid_smtid_docids[qid])
                tgt_qid_smtid_docids[qid][tgt_smtid] = tgt_smtid_to_docids[tgt_smtid]

                tgt_smtid_total_num += 1 
                docids_total_num += len(tgt_smtid_to_docids[tgt_smtid])

    print("ignore_num = {}, total_num = {}, ratio = {:.3f}".format(ignore_num, total_num, ignore_num/total_num))
    print("target_smtids num per src_smtid = {:.3f}".format(np.mean([len(xs) for xs in src_smtid_to_tgt_smtids.values()])))
    print("average number of target_smtids per query = {:.3f}, docids per query = {:.3f}".format(
        tgt_smtid_total_num / len(tgt_qid_smtid_docids), docids_total_num / len(tgt_qid_smtid_docids)
    ))

    with open(os.path.join(out_dir, "qid_smtid_docids.train.json"), "w") as fout:
        ujson.dump(tgt_qid_smtid_docids, fout)