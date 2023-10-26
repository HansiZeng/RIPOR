import ujson 
import os 
from tqdm import tqdm 

nway=50
in_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/"
teacher_score_path = os.path.join(in_dir, "qrel_added_qid_docids_teacher_scores.train.json")
out_path = os.path.join(in_dir, f"teacher_rerank_{nway}_way.train.json")

with open(out_path, "w") as fout:
    with open(teacher_score_path) as fin:
        for line in tqdm(fin, total=1_000_000):
            example = ujson.loads(line)
            qid = example["qid"]
            docids = example["docids"][:nway]
            scores = example["scores"][:nway]

            assert scores[1] >= scores[2]
            assert scores[3] >= scores[4]
            assert len(docids) >= nway, (len(docids), nway)

            labels = [1/(x+1) for x in range(5)]
            labels += [0. for x in range(5)]
            labels += [-0.5 for x in range(nway-10)]

            assert len(labels) == nway, (len(labels), nway)

            new_example = {"qid": qid, "docids": docids, "labels": labels}
            fout.write(ujson.dumps(new_example) + "\n")


