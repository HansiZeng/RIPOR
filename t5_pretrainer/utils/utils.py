import os
import torch.distributed as dist
import random
import ujson

def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def get_dataset_name(path):
    # small (hard-coded !) snippet to get a dataset name from a Q_COLLECTION_PATH or a EVAL_QREL_PATH (full paths)
    if "TREC_DL_2019" in path:
        return "TREC_DL_2019"
    elif "trec2020" in path or "TREC_DL_2020" in path:
        return "TREC_DL_2020"
    elif "msmarco" in path:
        if "train_queries" in path:
            return "MSMARCO_TRAIN"
        else:
            return "MSMARCO"
    elif "MSMarco-v2" in path:
        if "dev_1" in path:
            return "MSMARCO_v2_dev1"
        else:
            assert "dev_2" in path
            return "MSMARCO_v2_dev2"
    elif "toy" in path:
        return "TOY"
    elif "nq-320k" in path:
        return "NQ_320K"
    else:
        return "other_dataset"
    
def flatten_list(nested_list):
    out_list = []
    for cur_list in nested_list:
        for elem in cur_list:
            assert type(elem) != list, elem
            out_list.append(elem)
            
    return out_list

def convert_ptsmtids_to_strsmtid(input_smtids, seq_length):
    assert input_smtids.dim() == 3, input_smtids.dim()
    assert input_smtids.size(2) == seq_length + 1, (input_smtids.size(1), seq_length)
    #print(input_smtids[0][0])

    input_smtids = input_smtids.cpu().tolist()
    out_list = []
    for beam_smtids in input_smtids:
        beam_list = []
        for smtids in beam_smtids:
            beam_list.append("_".join([str(x) for x in smtids[1:]]))
        out_list.append(beam_list)
    
    return out_list

def form_strsmtid_from_prefix_and_lastsmtids(prefix_smtid, last_smtids, last_smtids_scores):
    assert isinstance(prefix_smtid, list) and isinstance(last_smtids, list), (prefix_smtid, last_smtids)
    
    all_strsmtids, all_scores = [], []
    for prefix, lasts, lasts_scores in zip(prefix_smtid, last_smtids, last_smtids_scores):
        strsmtids, scores = [], []    
        for last, score in zip(lasts, lasts_scores):
            strsmtid = "_".join([str(p) for p in prefix]) + "_" + str(last)
            strsmtids.append(strsmtid)
            scores.append(score)
        all_strsmtids.append(strsmtids)
        all_scores.append(scores)
    
    return all_strsmtids, all_scores

def partition_fn(lst, num_partitions):
    if len(lst) < num_partitions:
        raise ValueError(f"list size is {len(lst)} which is smaller num_partitions: {num_partitions}")
    partition_size = len(lst) // num_partitions
    partitions = [lst[i*partition_size:(i+1)*partition_size] for i in range(num_partitions)]
    # If the number of elements does not divide evenly into the number of partitions,
    # the last partition will take all remaining elements
    partitions[-1].extend(lst[num_partitions*partition_size:])
    return partitions

def sample_from_partitions(lst, num_partitions, num_samples):
    partitions = partition_fn(lst, num_partitions)
    num_samples_per_partition = num_samples // num_partitions
    remainder = num_samples % num_partitions

    samples = []

    for i, partition in enumerate(partitions):
        if i < remainder:
            # If there is a remainder, add one more sample to the first 'remainder' partitions
            samples.extend(random.sample(partition, num_samples_per_partition + 1))
        else:
            samples.extend(random.sample(partition, num_samples_per_partition))

    return samples


def from_qrel_to_qsmtid_rel(docid_to_smtid_path, qrels_path, truncate_smtid):
    with open(docid_to_smtid_path) as fin:
        docid_to_smtid = ujson.load(fin)

    if not truncate_smtid:
        docid_to_strsmtid = {}
        for docid, smtid in docid_to_smtid.items():
            assert smtid[0] == -1, smtid 
            str_smtid = "_".join([str(x) for x in smtid[1:]])
            docid_to_strsmtid[docid] = str_smtid 
    else:
        print("Warning: we truncate the smtid for smtid_tree evaluation")
        all_smtids = [list(xs) for xs in docid_to_smtid.values()]
        smtid_lengths = [len(xs) for xs in all_smtids]
        max_new_tokens = max(smtid_lengths) - 2  # -1 and eos_token_id is added 

        docid_to_strsmtid = {}
        for docid, smtid in docid_to_smtid.items():
            assert smtid[0] == -1, smtid 
            str_smtid = "_".join([str(x) for x in smtid[1:1+max_new_tokens]])
            #print(str_smtid)
            docid_to_strsmtid[docid] = str_smtid 
    
    with open(qrels_path) as fin:
        qrel_data = ujson.load(fin)
    qid_to_relsmtid_data = {}
    for qid in qrel_data:
        qid_to_relsmtid_data[qid] = {}
        for docid, s in qrel_data[qid].items():
            rel_smtid = docid_to_strsmtid[docid]
            qid_to_relsmtid_data[qid][rel_smtid] = s 

    return qid_to_relsmtid_data