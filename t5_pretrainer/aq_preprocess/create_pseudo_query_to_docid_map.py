import ujson 
import os 
from tqdm import tqdm

docid_to_rankdata_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/pid_qids_rerank_scores.json"
pseudo_queries_path = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/pseudo_queries.all.tsv"
out_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/"

with open(docid_to_rankdata_path) as fin:
    docid_to_rankdata = ujson.load(fin)

qid_to_query = {}
with open(pseudo_queries_path) as fin:
    for line in fin:
        qid, query = line.strip().split("\t")
        qid_to_query[qid] = query 

train_examples = []
filtered_train_examples = []
for docid, rankdata in tqdm(docid_to_rankdata.items(), total=len(docid_to_rankdata)):
    for qid, score in rankdata.items():
        query = qid_to_query[qid]
        train_examples.append({"docid": docid, "query": query})
        if score > 1.0:
            filtered_train_examples.append({"docid": docid, "query": query})

print("size of train_examples = {}, filtered_examples = {}, ratio = {:.3f}".format(len(train_examples), len(filtered_train_examples),
                                                                                   len(filtered_train_examples)/len(train_examples)))

with open(os.path.join(out_dir, "query_to_docid.train.json"), "w") as fout:
    for example in train_examples:
        fout.write(ujson.dumps(example) + "\n")

with open(os.path.join(out_dir, "query_to_docid.train.filtered.json"), "w") as fout:
    for example in filtered_train_examples:
        fout.write(ujson.dumps(example) + "\n")