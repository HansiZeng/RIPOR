#!/bin/bash

experiment_dir=experiments-msmarco-t5seq-cross-encoder

data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/
#collection_path=$data_root_dir/full_collection/raw.tsv
queries_path=$data_root_dir/train_queries/labeled_queries/raw.tsv
output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json

model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5seq_aq_encoder_seq2seq_1_qrf_mgmse_s2s_self_mnt_32_dcy_2"
pretrained_path=$model_dir/checkpoint

# might need to change for every experiment
neg_sample=50
lr=1e-4
for ep in 1 5 
do

        bce_example_path=$data_dir/ce_train/bce_examples."$neg_sample"_"$neg_sample".train.tsv 
        run_name=t5seq_cross_encoder_"$neg_sample"_"$neg_sample"_"$lr"_"$ep"

        # also need to be changed by condition

        python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
                --epochs=$ep \
                --run_name=$run_name \
                --learning_rate=$lr \
                --loss_type=t5seq_bce \
                --model_type=t5seq_cross_encoder \
                --bce_example_path=$bce_example_path \
                --output_dir=$output_dir \
                --task_names='["cls"]' \
                --wandb_project_name=t5seq_cross_encoder \
                --use_fp16 \
                --max_length=128 \
                --per_device_train_batch_size=128 \
                --queries_path=$queries_path \
                --save_steps=50000 \
                --docid_to_smtid_path=$docid_to_smtid_path \
                --pretrained_path=$pretrained_path \
                --warmup_ratio=0.05
done