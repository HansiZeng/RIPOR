#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-full-16-1024-t5seq-aq


# seq2seq_0
task=t5seq_aq_encoder_seq2seq
query_to_docid_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/query_to_docid.train.json
data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

# need to change for every experiment
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
pretrained_path=$model_dir/no_share_checkpoint/
run_name=t5seq_aq_encoder_seq2seq_0

# train
#python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
#        --max_steps=200_000 \
#        --run_name=$run_name  \
#        --learning_rate=1e-3 \
#        --loss_type=$task \
#        --model_name_or_path=t5-base \
#        --model_type=t5_docid_gen_encoder \
#        --per_device_train_batch_size=256 \
#        --pretrained_path=$pretrained_path \
#        --query_to_docid_path=$query_to_docid_path \
#        --docid_to_smtid_path=$docid_to_smtid_path \
#        --output_dir=$output_dir \
#        --save_steps=50_000 \
#        --task_names='["rank"]' \
#        --wandb_project_name=full_16_1024_t5seq_encoder \
#        --use_fp16 \
#        --warmup_ratio=0.045

# seq2seq_1
task=t5seq_aq_encoder_margin_mse
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

# need to change for every experiment
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_0"
pretrained_path=$model_dir/checkpoint/
run_name=t5seq_aq_encoder_seq2seq_1

# also need to be changed by condition
teacher_score_path=$data_dir/out/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=120 \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=t5seq_aq_encoder_margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_docid_gen_encoder \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=full_16_1024_t5seq_encoder \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=64 \
        --per_device_train_batch_size=128 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path \
        --docid_to_smtid_path=$docid_to_smtid_path


python -m t5_pretrainer.aq_preprocess.change_customized_embed_layer --model_dir=/home/ec2-user/quic-efs/user/hansizeng/work$
t5_pretrainer/t5_pretrainer/experiments-full-16-1024-t5seq-aq/t5_docid_gen_encoder_1/ --K=1024