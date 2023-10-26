#!/bin/bash
train_queries_path="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries/raw.tsv"
collection_path=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/full_collection

# need to change every time
max_new_token=16
data_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1
qid_smtid_docids_path=$data_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.rerank \
    --train_queries_path=$train_queries_path \
    --collection_path=$collection_path \
    --model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --max_length=256 \
    --batch_size=256 \
    --qid_smtid_docids_path=$qid_smtid_docids_path \
    --task=cross_encoder_rerank_for_qid_smtid_docids

python -m t5_pretrainer.rerank \
    --out_dir=$data_dir/sub_smtid_"$max_new_token"_out \
    --task=cross_encoder_rerank_for_qid_smtid_docids_2