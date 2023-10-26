#!/bin/bash

task=all_aq_pipline
data_root_dir=/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full
collection_path=$data_root_dir/full_collection/
q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/queries_2019/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/queries_2020/","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_queries/"]'
eval_qrel_path='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/dev_qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2019/qrel_binary.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel.json","/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/TREC_DL_2020/qrel_binary.json"]'
experiment_dir=experiments-full-16-1024-t5seq-aq

if [ $task = all_aq_pipline ]; then 
    echo "task: $task"

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    pretrained_path=$model_dir/checkpoint
    index_dir=$model_dir/aq_index
    mmap_dir=$model_dir/mmap
    out_dir=$model_dir/aq_out

    M=16
    nbits=10
    K=$((2 ** $nbits))
    echo M: $M nbits: $nbits K: $K
    echo $model_dir

    #python -m torch.distributed.launch --nproc_per_node=8 -m t5_pretrainer.evaluate \
    #--pretrained_path=$pretrained_path \
    #--index_dir=$mmap_dir \
    #--task=mmap \
    #--encoder_type=t5seq_pretrain_encoder \
    #--collection_path=$collection_path

    #python -m t5_pretrainer.evaluate \
    #--task=mmap_2 \
    #--index_dir=$mmap_dir \
    #--mmap_dir=$mmap_dir

    python -m t5_pretrainer.evaluate \
    --task=aq_index \
    --num_subvectors_for_pq=$M \
    --codebook_bits=$nbits \
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
    --M=$M \
    --bits=$nbits

    python -m t5_pretrainer.aq_preprocess.change_customized_embed_layer \
    --model_dir=$model_dir \
    --K=$K
elif [ $task = aq_to_flat_index_search_evaluate ]; then 
    echo "task: $task"
    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
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
    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    index_dir=$model_dir/index
    out_dir=$model_dir/out/
    pretrained_path=$model_dir/checkpoint

    python -m t5_pretrainer.evaluate \
    --task=retrieve \
    --pretrained_path=$pretrained_path \
    --index_dir=$index_dir \
    --out_dir=$out_dir  \
    --q_collection_paths='["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries"]' \
    --topk=100 \
    --encoder_type=t5seq_pretrain_encoder
elif [ $task = all_pipline ]; then 
    echo "task: $task"

    model_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
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
    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

    model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1
    pretrained_path=$model_dir/checkpoint
    train_query_dir="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-full/all_train_queries/train_queries/"

    # need to remove later
    max_new_token=8

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
elif [ $task = "t5seq_aq_retrieve_docids_use_sub_smtid" ]; then 
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "task: $task"
    data_dir="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5_docid_gen_encoder_1"
    docid_to_smtid_path=$data_dir/aq_smtid/docid_to_smtid.json 

    # need to modify for a new experiment
    max_new_token=8
    model_dir=/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/$experiment_dir/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_8_dcy_2
    pretrained_path=$model_dir/checkpoint
    out_dir=$model_dir/out_docid_from_sub_"$max_new_token"_top1000/

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
else 
echo "Error: Unknown task."
exit 1
fi