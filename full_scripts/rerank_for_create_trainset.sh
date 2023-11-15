#!/bin/bash

data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
#run_path="./data/msmarco-full/bm25_run/top100.marco.train.all.json"
#out_dir="./data/msmarco-full/bm25_run/"

root_dir=./experiments-full-t5seq-aq/
run_path=$root_dir/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/run.json
out_dir=$root_dir/t5_docid_gen_encoder_1/out/MSMARCO_TRAIN/
q_collection_path=./data/msmarco-full/all_train_queries/train_queries


python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset \
    --run_json_path=$run_path \
    --out_dir=$out_dir \
    --collection_path=$collection_path \
    --q_collection_path=$q_collection_path \
    --json_type=json \
    --batch_size=256

python -m t5_pretrainer.rerank \
    --task=rerank_for_create_trainset_2 \
    --out_dir=$out_dir
