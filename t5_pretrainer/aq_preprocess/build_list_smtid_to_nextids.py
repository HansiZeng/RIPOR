import os 
import argparse
import ujson
from tqdm import tqdm
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docid_to_smtid_path", default=None, type=str)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    list_smtid_to_nextids_path = os.path.join(os.path.dirname(args.docid_to_smtid_path), "list_smtid_to_nextids.pkl")
    if os.path.exists(list_smtid_to_nextids_path):
        print(f"{list_smtid_to_nextids_path} exists")
    else:
        # create prefix_constrain logit processor
        with open(args.docid_to_smtid_path) as fin:
            docid_to_smtids = ujson.load(fin)
        list_smtid_to_nextids = [dict() for _ in range(len(docid_to_smtids["0"])-1)]
        for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
            for i in range(len(smtids)-1):
                cur_smtid = "_".join([str(x) for x in smtids[:i+1]])
                next_id = int(smtids[i+1])

                cur_dict = list_smtid_to_nextids[i]
                if cur_smtid not in cur_dict:
                    cur_dict[cur_smtid] = {next_id}
                else:
                    cur_dict[cur_smtid].add(next_id)
        for i, cur_dict in enumerate(list_smtid_to_nextids):
            for cur_smtid in cur_dict:
                cur_dict[cur_smtid] = list(cur_dict[cur_smtid])
            print(f"{i}-th step has {len(cur_dict):,} effective smtid ")

        print("save list_smtid_to_nextids")
        with open(list_smtid_to_nextids_path, "wb") as fout:
            pickle.dump(list_smtid_to_nextids, fout)