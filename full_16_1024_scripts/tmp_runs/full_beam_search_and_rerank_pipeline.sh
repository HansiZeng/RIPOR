#!/bin/bash

task=all_aq_pipline
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-16-1024-t5seq-aq

task="t5seq_aq_get_qid_to_smtid_rankdata"
echo "task: $task"
data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1
pretrained_path=$model_dir/checkpoint
train_query_dir="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries/"

train_queries_path="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries/raw.tsv"

# need to remove later
for max_new_token in 4 8
do
    out_dir=$model_dir/sub_smtid_"${max_new_token}"_out/
    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.evaluate \
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

   python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
    --root_dir=$out_dir
done

for max_new_token in 4 8
do 
    qid_smtid_docids_path=$model_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.rerank \
        --train_queries_path=$train_queries_path \
        --collection_path=$collection_path \
        --model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --max_length=256 \
        --batch_size=256 \
        --qid_smtid_docids_path=$qid_smtid_docids_path \
        --task=cross_encoder_rerank_for_qid_smtid_docids

    python -m t5_pretrainer.rerank \
        --out_dir=$model_dir/sub_smtid_"$max_new_token"_out \
        --task=cross_encoder_rerank_for_qid_smtid_docids_2
done