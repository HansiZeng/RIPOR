#!/bin/bash

task=t5seq_aq_retrieve_docids_use_sub_smtid
data_root_dir=./data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["./data/msmarco-full/TREC_DL_2019/queries_2019/","./data/msmarco-full/TREC_DL_2020/queries_2020/","./data/msmarco-full/dev_queries/"]'
eval_qrel_path='["./data/msmarco-full/dev_qrel.json","./data/msmarco-full/TREC_DL_2019/qrel.json","./data/msmarco-full/TREC_DL_2019/qrel_binary.json","./data/msmarco-full/TREC_DL_2020/qrel.json","./data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-t5seq-aq

if [ $task = all_aq_pipline ]; then 
    echo "task: $task"

    model_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/aq_index
    mmap_dir=$model_dir/mmap
    out_dir=$model_dir/aq_out

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$mmap_dir \
    --task=mmap \
    --encoder_type=t5seq_pretrain_encoder \
    --collection_path=$collection_path

    python -m t5_pretrainer.evaluate \
    --task=mmap_2 \
    --index_dir=$mmap_dir \
    --mmap_dir=$mmap_dir

    python -m t5_pretrainer.evaluate \
    --task=aq_index \
    --num_subvectors_for_pq=32 \
    --index_dir=$index_dir \
    --mmap_dir=$mmap_dir

    python -m t5_pretrainer.evaluate \
    --task=aq_evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --q_collection_paths=$q_collection_paths \
    --eval_qrel_path=$eval_qrel_path  \
    --mmap_dir=$mmap_dir

    python t5_pretrainer/aq_preprocess/create_customized_smtid_file.py \
        --model_dir=$model_dir \
        --M=32 \
        --bits=8

    python -m t5_pretrainer.aq_preprocss.change_embed_layer \
        --model_dir=$model_dir

elif [ $task = aq_to_flat_index_search_evaluate ]; then 
    echo "task: $task"
    data_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json

    model_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/no_share_checkpoint
    index_dir=$model_dir/aq_flat_index
    out_dir=$model_dir/aq_flat_out

    python -m t5_pretrainer.evaluate --pretrained_path=$pretrained_path \
        --collection_path=$collection_path/raw.tsv \
        --docid_to_smtid_path=$docid_to_smtid_path \
        --index_dir=$index_dir \
        --out_dir=$out_dir  \
        --task=$task \
        --eval_qrel_path=$eval_qrel_path
elif [ $task == "retrieve_train_queries" ]; then 
    echo "run retrieve_train_queries task"

    # the model_dir should be changed every time
    experiment_dir=experiments-full-t5seq-aq
    model_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    index_dir=$model_dir/index
    out_dir=$model_dir/out/
    pretrained_path=$model_dir/checkpoint

    python -m t5_pretrainer.evaluate \
    --task=retrieve \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir  \
    --q_collection_paths='["./data/msmarco-full/all_train_queries/train_queries"]' \
    --topk=100 \
    --encoder_type=t5seq_pretrain_encoder
elif [ $task = all_pipline ]; then 
    echo "task: $task"

    model_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/index
    out_dir=$model_dir/out

    python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.evaluate \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --task=index \
    --encoder_type=t5seq_pretrain_encoder \
    --collection_path=$collection_path

    python -m t5_pretrainer.evaluate \
    --task=index_2 \
    --index_dir=$index_dir 

    python -m t5_pretrainer.evaluate \
    --task=retrieve \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir \
    --encoder_type=t5seq_pretrain_encoder \
    --q_collection_paths=$q_collection_paths \
    --eval_qrel_path=$eval_qrel_path
elif [ $task = "t5seq_aq_get_qid_to_smtid_rankdata" ]; then 
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "task: $task"
    data_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

    model_dir=./$experiment_dir/t5seq_aq_encoder_seq2seq_1
    pretrained_path=$model_dir/checkpoint
    train_query_dir="./data/msmarco-full/all_train_queries/train_queries/"

    # Apply beam search to generate prefix with length 4, 8, 16
    for max_new_token in 4 8 16
    do
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

        python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
            --root_dir=$out_dir
    done

    # Since prefix=32 and prefix=16 almost corresponds to the same doc. To save time, we directly expand it from 16 to 32.
    python t5_pretrainer/aq_preprocess/expand_smtid_for_qid_smtid_docids.py \
        --data_dir=$data_dir \
        --src_qid_smtid_rankdata_path=$model_dir/sub_smtid_16_out/qid_smtid_rankdata.json \
        --out_dir=$model_dir/sub_smtid_32_out

    python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
            --root_dir=$model_dir/sub_smtid_32_out

    # let's rerank the data 
    for max_new_token in 4 8 16 32
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
elif [ $task = "t5seq_aq_retrieve_docids_use_sub_smtid" ]; then 
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "task: $task"
    data_dir="./$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

    # need to modify for a new experiment
    max_new_token=32
    model_dir=./$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/
    pretrained_path=$model_dir/checkpoint
    out_dir=$model_dir/out_docid_from_sub_"$max_new_token"_top1000/

    python -m t5_pretrainer.aq_preprocess.build_list_smtid_to_nextids \
        --docid_to_smtid_path=$docid_to_smtid_path

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
elif [ $task = "sanity_check_t5seq_aq_get_qid_to_smtid_rankdata" ]; then 
    task=t5seq_aq_get_qid_to_smtid_rankdata
    #export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "task: $task"
    experiment_dir=/gypsum/work1/zamani/hzeng/RIPOR/experiments-full-t5seq-aq/
    data_dir=$experiment_dir/t5_docid_gen_encoder_1
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

    model_dir=$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2
    pretrained_path=$model_dir/checkpoint
    train_query_dir="../data/msmarco-full/all_train_queries/train_queries/"

    # Apply beam search to generate prefix with length 4, 8, 16
    for max_new_token in 4 
    do
        out_dir=$model_dir/sub_smtid_"${max_new_token}"_out/
        python -m torch.distributed.launch --nproc_per_node=1 -m t5_pretrainer.evaluate \
            --pretrained_path=$pretrained_path \
            --out_dir=$out_dir \
            --task=$task \
            --docid_to_smtid_path=$docid_to_smtid_path \
            --topk=100 \
            --batch_size=4 \
            --train_query_dir=$train_query_dir \
            --max_new_token=$max_new_token

        #python -m t5_pretrainer.evaluate \
        #    --task="$task"_2 \
        #    --out_dir=$out_dir 

        #python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
        #    --root_dir=$out_dir
    done

    # Since prefix=32 and prefix=16 almost corresponds to the same doc. To save time, we directly expand it from 16 to 32.
    #python t5_pretrainer/aq_preprocess/expand_smtid_for_qid_smtid_docids.py \
    #    --data_dir=$data_dir \
    #    --src_qid_smtid_rankdata_path=$model_dir/sub_smtid_16_out/qid_smtid_rankdata.json \
    #    --out_dir=$model_dir/sub_smtid_32_out

    #python t5_pretrainer/aq_preprocess/argparse_from_qid_smtid_rank_to_qid_smtid_docids.py \
    #        --root_dir=$model_dir/sub_smtid_32_out

    # let's rerank the data 
    #for max_new_token in 4 8 16 32
    #do 
    #    qid_smtid_docids_path=$model_dir/sub_smtid_"$max_new_token"_out/qid_smtid_docids.train.json

    #   python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.rerank \
    #        --train_queries_path=$train_queries_path \
    #        --collection_path=$collection_path \
    #        --model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    #        --max_length=256 \
    #        --batch_size=256 \
    #        --qid_smtid_docids_path=$qid_smtid_docids_path \
    #        --task=cross_encoder_rerank_for_qid_smtid_docids

    #    python -m t5_pretrainer.rerank \
    #        --out_dir=$model_dir/sub_smtid_"$max_new_token"_out \
    #        --task=cross_encoder_rerank_for_qid_smtid_docids_2
    #done
else 
echo "Error: Unknown task."
exit 1
fi