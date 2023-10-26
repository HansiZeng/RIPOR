#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

task=t5seq_aq_retrieve_docids_use_sub_smtid
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

# need to modify for a new experiment
model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1/
pretrained_path=$model_dir/checkpoint

for max_new_token in 2 4 6 8 10 12 14 16
do
    out_dir=$model_dir/out_docid_from_sub_"$max_new_token"_top1000/
    echo max new token $max_new_token

    python -m torch.distributed.launch --nproc_per_node=4 -m t5_pretrainer.evaluate \
        --pretrained_path=$pretrained_path \
        --out_dir=$out_dir \
        --task=t5seq_aq_retrieve_docids \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --q_collection_paths=$q_collection_paths \
        --batch_size=1 \
        --max_new_token_for_docid=$max_new_token \
        --topk=1000

    python -m t5_pretrainer.evaluate \
        --task=t5seq_aq_retrieve_docids_2 \
        --out_dir=$out_dir \
        --q_collection_paths=$q_collection_paths \
        --eval_qrel_path=$eval_qrel_path
done