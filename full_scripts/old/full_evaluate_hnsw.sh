#!/bin/bash

task=all_pipline

data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

if [ $task = "all_pipline" ]; then 
    echo "task: $task"

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/hnsw_index
    mmap_dir=$model_dir/mmap
    out_dir=$model_dir/hnsw_out

    python -m t5_pretrainer.evaluate \
    --task=hnsw_index \
    --index_dir=$index_dir \
    --mmap_dir=$mmap_dir

    python -m t5_pretrainer.evaluate \
    --task=hnsw_evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --q_collection_paths=$q_collection_paths \
    --eval_qrel_path=$eval_qrel_path  \
    --mmap_dir=$mmap_dir
else 
echo "Error: Unknown task."
exit 1
fi