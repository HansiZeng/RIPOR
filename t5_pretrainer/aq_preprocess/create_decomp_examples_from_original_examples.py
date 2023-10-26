import ujson 
import os 

# assume we handle with the case of decay = 2
score_to_early_score = {8: 0.5/0.75, 16: 0.75/0.875, 32: 0.875/1.0}
mnt_to_recover_factor = {8: 1/0.75, 16: 1/0.875, 32: 1.}
mnt_to_factors = {8: [0.5, 0.25],
                  16: [0.5, 0.25, 0.125],
                  32: [0.5, 0.25, 0.125, 0.125]}

max_new_token = 8
recover_factor = mnt_to_recover_factor[max_new_token]
factors = mnt_to_factors[max_new_token]
print("max_new_token: ", max_new_token, "recover_factor: ", recover_factor, 
      "factors: ", factors)

root_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1/"
source_example_path = os.path.join(root_dir, f"sub_smtid_train_decay2/qid_smtids_scores_{max_new_token}.train.json")
out_dir = os.path.join(root_dir, "decomp_sub_smtid_train_decay2")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

with open(source_example_path) as fin:
    with open(os.path.join(out_dir, f"decomp_qid_smtids_scores_{max_new_token}.train.json"), "w") as fout:
        for line in fin:
            example = ujson.loads(line)

            decomp_scores_list = []
            for score in example["scores"]:
                rc_score = score * recover_factor
                dp_scores = [rc_score * r for r in factors]
                decomp_scores_list.append(dp_scores)

            example["decomp_scores"] = decomp_scores_list
            del example["scores"]

            fout.write(ujson.dumps(example) + "\n")

