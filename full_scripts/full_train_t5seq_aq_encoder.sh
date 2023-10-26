#!/bin/bash
task=t5seq_aq_encoder_lng_knp_margin_mse_and_seq2seq
experiment_dir=experiments-full-t5seq-aq
if [ $task = "t5seq_aq_encoder_margin_mse" ]; then
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
            --epochs=150 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names='["rank"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --docid_to_smtid_path=$docid_to_smtid_path
elif [ $task = t5seq_aq_encoder_seq2seq ]; then 
    query_to_docid_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/query_to_docid.train.json
    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/no_share_checkpoint/
    run_name=t5seq_aq_encoder_seq2seq_0

    # train
    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --max_steps=250_000 \
            --run_name=$run_name  \
            --learning_rate=1e-3 \
            --loss_type=$task \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --per_device_train_batch_size=256 \
            --pretrained_path=$pretrained_path \
            --query_to_docid_path=$query_to_docid_path \
            --docid_to_smtid_path=$docid_to_smtid_path \
            --output_dir=$output_dir \
            --save_steps=50_000 \
            --task_names='["rank"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --warmup_ratio=0.045
elif [ $task = "t5seq_aq_encoder_ranknet" ]; then
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    teacher_rerank_nway_path=$data_dir/out/MSMARCO_TRAIN/teacher_rerank_50_way.train.json

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_0"
    pretrained_path=$model_dir/checkpoint/
    run_name=t5seq_aq_ranknet_0
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=8 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_ranknet \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_rerank_nway_path=$teacher_rerank_nway_path \
            --output_dir=$output_dir \
            --task_names='["rank"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=32 \
            --per_device_train_batch_size=8 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --docid_to_smtid_path=$docid_to_smtid_path \
            --warmup_ratio=0.045
elif [ $task = "t5seq_aq_encoder_margin_mse_sub_smtid" ]; then
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_mnt_4_dcy_2"
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    decay=2
    max_new_token=8
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1/
    teacher_score_path=$teacher_score_dir/sub_smtid_train_decay"$decay"/qid_smtids_scores_"$max_new_token".train.json
    run_name=t5seq_aq_encoder_seq2seq_1_mnt_"$max_new_token"_dcy_"$decay"_b

    echo $teacher_score_path

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=150 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names='["rank"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid

elif [ $task = "t5seq_aq_encoder_knp_margin_mse_sub_smtid" ]; then
    echo "task: $task"
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_knp_mnt_32_dcy_2"
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    decay=2
    max_new_token=32
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_knp_mnt_32_dcy_2
    teacher_score_path=$teacher_score_dir/knp_sub_smtid_train_decay"$decay"/knp_qid_smtids_scores_"$max_new_token"_top_80.train.json
    run_name=t5seq_aq_encoder_seq2seq_1_knp_self_mnt_16_"$max_new_token"_dcy_"$decay"

    echo $teacher_score_path

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=120 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_knp_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names='["rank","early_rank"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid
elif [ $task = "t5seq_aq_encoder_lng_knp_margin_mse_sub_smtid" ]; then
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2"
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    decay=2
    max_new_token=32
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/
    teacher_score_path=$teacher_score_dir/qrel_first_sub_smtid_train_decay"$decay"/lng_knp_qid_smtids_scores_"$max_new_token"_top_100.train.json
    run_name=t5seq_aq_encoder_seq2seq_1_qrf_lng_knp_self_mnt_"$max_new_token"_dcy_"$decay"

    if [ $max_new_token -eq 16 ]; then
        task_names='["rank","rank_8"]'
    elif [ $max_new_token -eq 32 ]; then 
        task_names='["rank","rank_8","rank_16"]'
    else 
        echo "Error: Unknown task_names."
        exit 1
    fi
    echo $teacher_score_path

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=120 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_lng_knp_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names=$task_names \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid
elif [ $task = "t5seq_aq_encoder_lng_knp_margin_mse_and_seq2seq" ]; then
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2"
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    max_new_token=32
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2/
    teacher_score_path=$teacher_score_dir/qrel_first_sub_smtid_train_decay2/lng_knp_qid_smtids_scores_"$max_new_token"_top_100.train.json
    run_name=t5seq_aq_encoder_seq2seq_1_qrf_mgmse_s2s_self_mnt_"$max_new_token"_dcy_2

    echo $teacher_score_path

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=120 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=$task \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names='["rank","rank_8","rank_16","seq","seq_8","seq_16"]' \
            --wandb_project_name=1M_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid
elif [ $task = "t5seq_aq_encoder_decomp_margin_mse_sub_smtid" ]; then
    data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
    collection_path=$data_root_dir/full_collection/
    queries_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/doc2query/all_train_queries/train_queries

    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json
    output_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/"

    # need to change for every experiment
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_mnt_4_dcy_2"
    pretrained_path=$model_dir/checkpoint/

    # also need to be changed by condition
    decay=2
    max_new_token=8
    teacher_score_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1/
    teacher_score_path=$teacher_score_dir/decomp_sub_smtid_train_decay"$decay"/decomp_qid_smtids_scores_"$max_new_token".train.json
    run_name=seq2seq_1_decomp_mnt_"$max_new_token"_dcy_"$decay"

    echo $teacher_score_path

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.main \
            --epochs=120 \
            --run_name=$run_name \
            --learning_rate=1e-4 \
            --loss_type=t5seq_aq_encoder_decomp_margin_mse \
            --model_name_or_path=t5-base \
            --model_type=t5_docid_gen_encoder \
            --teacher_score_path=$teacher_score_path \
            --output_dir=$output_dir \
            --task_names='["rank_4","rank_8"]' \
            --wandb_project_name=full_t5seq_encoder \
            --use_fp16 \
            --collection_path=$collection_path \
            --max_length=64 \
            --per_device_train_batch_size=128 \
            --queries_path=$queries_path \
            --pretrained_path=$pretrained_path \
            --smtid_as_docid
else 
echo "Error: Unknown task."
exit 1
fi