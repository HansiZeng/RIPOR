import os 
import ujson 


root_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/"
if "full" in root_dir:
    trunc_num = 100
elif "1M" in root_dir:
    trunc_num = 80
else:
    raise ValueError("root_dir not predefined.")
print("trunc_num: ", trunc_num)

src_example_path = os.path.join(root_dir, "qrel_first_sub_smtid_train_decay2/lng_knp_qid_smtids_scores_32.train.json")
target_example_path = os.path.join(root_dir, f"qrel_first_sub_smtid_train_decay2/lng_knp_qid_smtids_scores_32_top_{trunc_num}.train.json")

with open(src_example_path) as fin:
    with open(target_example_path, "w") as fout:
        for i, line in enumerate(fin):
            example = ujson.loads(line)
            if i == 0:
                print(example.keys())
            fout.write(ujson.dumps({
                "qid": example["qid"],
                "smtids": example["smtids"][:trunc_num],
                "smtid_8_scores": example["smtid_8_scores"][:trunc_num],
                "smtid_16_scores": example["smtid_16_scores"][:trunc_num],
                "scores": example["scores"][:trunc_num],
            }) + "\n")
    