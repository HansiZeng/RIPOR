import os 
import ujson 

trunc_num = 80

root_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_knp_mnt_32_dcy_2/"
src_example_path = os.path.join(root_dir, "knp_sub_smtid_train_decay2/knp_qid_smtids_scores_32.train.json")
target_example_path = os.path.join(root_dir, f"knp_sub_smtid_train_decay2/knp_qid_smtids_scores_32_top_{trunc_num}.train.json")

with open(src_example_path) as fin:
    with open(target_example_path, "w") as fout:
        for i, line in enumerate(fin):
            example = ujson.loads(line)
            if i == 0:
                print(example.keys())
            fout.write(ujson.dumps({
                "qid": example["qid"],
                "smtids": example["smtids"][:trunc_num],
                "early_scores": example["early_scores"][:trunc_num],
                "scores": example["scores"][:trunc_num]
            }) + "\n")
    