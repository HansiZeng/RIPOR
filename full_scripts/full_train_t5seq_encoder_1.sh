#!/bin/bash

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
queries_path=./data/msmarco-full/all_train_queries/train_queries

# model dir
experiment_dir=experiments-full-t5seq-aq
model_dir="./$experiment_dir/t5_docid_gen_encoder_0"
pretrained_path=$model_dir/checkpoint/

# train_examples path
teacher_score_path=$model_dir/all_train/MSMARCO_TRAIN/qrel_added_qid_docids_teacher_scores.train.json
run_name=t5_docid_gen_encoder_1
output_dir="./$experiment_dir/"

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
        --epochs=50 \
        --run_name=$run_name \
        --learning_rate=1e-4 \
        --loss_type=t5seq_pretrain_margin_mse \
        --model_name_or_path=t5-base \
        --model_type=t5_docid_gen_encoder \
        --teacher_score_path=$teacher_score_path \
        --output_dir=$output_dir \
        --task_names='["rank"]' \
        --wandb_project_name=full_t5seq_encoder \
        --use_fp16 \
        --collection_path=$collection_path \
        --max_length=128 \
        --per_device_train_batch_size=64 \
        --queries_path=$queries_path \
        --pretrained_path=$pretrained_path 