#!/bin/bash
task=t5seq_aq_encoder_seq2seq
experiment_dir=experiments-full-16-1024-t5seq-aq

# when max_new_token=4
task=t5seq_aq_encoder_margin_mse_sub_smtid
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

# need to change for every experiment
model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1"
pretrained_path=$model_dir/checkpoint

# also need to be changed by condition
decay=2
max_new_token=4
teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1/
teacher_score_path=$teacher_score_dir/sub_smtid_train_decay"$decay"/qid_smtids_scores_"$max_new_token".train.json
run_name=t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_"$max_new_token"_dcy_"$decay"

echo $teacher_score_path

#python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
#        --epochs=120 \
#        --run_name=$run_name \
##        --learning_rate=1e-4 \
#        --loss_type=t5seq_aq_encoder_margin_mse \
#        --model_name_or_path=t5-base \
#        --model_type=t5_docid_gen_encoder \
#        --teacher_score_path=$teacher_score_path \
#        --output_dir=$output_dir \
#        --task_names='["rank"]' \
#        --wandb_project_name=full_16_1024_t5seq_encoder \
#        --use_fp16 \
#        --collection_path=$collection_path \
#        --max_length=64 \
#        --per_device_train_batch_size=128 \
#        --queries_path=$queries_path \
#        --pretrained_path=$pretrained_path \
#        --smtid_as_docid

# when max_new_token > 4
for max_new_token in 8 16
do  
    if [ $max_new_token -eq 8 ]; then 
        task_names='["rank","rank_4"]'
        prev_token=4
    elif [ $max_new_token -eq 16 ]; then
        task_names='["rank","rank_4","rank_8"]'
        prev_token=8
    elif [ $max_new_token -eq 32 ]; then 
        task_names='["rank","rank_4","rank_8","rank_16"]'
        prev_token=16
    else 
        echo "Error: Unknown task_names."
        exit 1
    fi
    echo $teacher_score_path

    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_"$prev_token"_dcy_2
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    decay=2
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1/
    teacher_score_path=$teacher_score_dir/lng_knp_sub_smtid_train_decay"$decay"/lng_knp_qid_smtids_scores_"$max_new_token".train.json
    run_name=t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_"$max_new_token"_dcy_"$decay"

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
    python -m torch.distributed.launch --nproc_per_node=7 -m t5_pretrainer.main \
            --epochs=100 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_lng_knp_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names=$task_names \
            --wandb_project_name=full_16_1024_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=96 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid
done