#!/bin/bash

task=t5seq_aq_get_qid_to_smtid_rankdata
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "task: $task"
data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1
pretrained_path=$model_dir/checkpoint
train_query_dir="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries/"

# need to remove later
max_new_token=16

out_dir=$model_dir/sub_smtid_"${max_new_token}"_out/
python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
    --pretrained_path=$pretrained_path \
    --out_dir=$out_dir \
    --task=$task \
    --docid_to_smtid_path=$docid_to_smtid_path \
    --topk=100 \
    --batch_size=4 \
    --train_query_dir=$train_query_dir \
    --max_new_token=$max_new_token

python -m t5_pretrainer.evaluate \
    --task="$task"_2 \
    --out_dir=$out_dir