#!/bin/bash
train_queries_path="./data/msmarco-full/all_train_queries/train_queries/raw.tsv"
collection_path=./data/msmarco-full/full_collection

# need to change every time
for max_new_token in 4 8 16 32
do
    data_dir=./experiments-full-lexical-ripor/t5_docid_gen_encoder_1/

    qid_smtid_docids_path=$data_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
done