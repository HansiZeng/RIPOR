#!/bin/bash
export CUDA_VISIBLE_DEVICES=4

experiment_dir="experiments-full-t5seq-aq"
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'

data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json

max_new_token=32
model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_mnt_32_dcy_2
pretrained_path=$model_dir/checkpoint
out_dir=$model_dir/rerank_out

python -m t5_pretrainer.evaluate \
    --task=t5seq_aq_rerank \
    --q_collection_paths=$q_collection_paths \
    --eval_qrel_path=$eval_qrel_path \
    --docid_to_smtid_path=$docid_to_smtid_path \
    --pretrained_path=$pretrained_path \
    --out_dir=$out_dir \
    --batch_size=256 \
    --max_length=64 \
    --max_new_token_for_docid=$max_new_token
    