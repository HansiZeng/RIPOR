from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os 
import ujson

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="t5-base")
    #apply_decoder_t5_stack: bool = field(default=False)
    #scaleup_output_hidden: bool = field(default=False)
    #shared_output_input_embeds: bool = field(default=True)
    subset_k: int = field(default=8)
    subset_tau: float = field(default=1.0)
    subset_hard: bool = field(default=False)
    subset_sample_docid: bool = field(default=False)

@dataclass
class Arguments:
    teacher_score_path: Optional[str] = field(
        default=None)
    collection_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/full_collection")
    queries_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/queries")
    qrels_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/qrels.json")
    output_dir: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments/")
    example_path: Optional[str] = field(default=None)
    pseudo_queries_to_docid_path: Optional[str] = field(default=None)
    pseudo_queries_to_mul_docid_path: Optional[str] = field(default=None)
    docid_to_smtid_path: Optional[str] = field(default=None)
    qid_to_smtid_path: Optional[str] = field(default=None)
    first_centroid_path: Optional[str] = field(default=None)
    second_centroid_path: Optional[str] = field(default=None)
    third_centroid_path: Optional[str] = field(default=None)
    qid_to_rrpids_path: Optional[str] = field(default=None)
    docid_decode_eval_path: Optional[str] = field(default=None)
    centroid_path: Optional[str] = field(default=None)
    centroid_idx: Optional[int] = field(default=None)
    triple_margin_mse_path: Optional[str] = field(default=None)
    query_to_docid_path: Optional[str] = field(default=None)
    teacher_rerank_nway_path: Optional[str] = field(default=None)
    bce_example_path: Optional[str] = field(default=None)

    run_name: str = field(default="t5_pretrainer_marginmse_tmp")
    pretrained_path: Optional[str] = field(default=None)
    loss_type: str = field(default="margin_mse")
    model_type: str = field(default="t5_docid_gen_encoder")
    do_eval: bool = field(default=False)

    max_length: int = field(default=256)
    learning_rate: float = field(default=1e-4)
    warmup_ratio: float = field(default=0.04)
    per_device_train_batch_size: int = field(default=32) 
    logging_steps: int = field(default=50)
    max_steps: int = field(default=-1)
    epochs: int = field(default=3)
    local_rank: int = field(default=-1)
    task_names: Optional[str] = field(default=None)
    #task_names: List[str] = field(default_factory=lambda: ["rank", "commit", "reg"]) # ["rank", "reg"] | ["rank", "commit", "reg"] | ["rank", "commit"] | ["rank"]
    ln_to_weight: Dict[str, float] = field(default_factory=lambda:   {"rank": 1.0, "commit": 1.0, "reg": 0.2}) # {"rank": 1.0, "reg": 0.002} | {"rank": 1.0, "commit": 1.0, "reg": 0.002} | {"rank": 1.0, "commit": 1.0} | {"rank": 1.0}
    multi_weights: Optional[List] = field(default=None)
    nway_label_type: Optional[str] = field(default=None)
    nway_rrpids: int = field(default=24)
    nway: int = field(default=12)
    eval_steps: int = field(default=50)
    use_fp16: bool = field(default=False)
    multi_vocab_sizes: bool = field(default=False)
    save_steps: int = field(default=15_000)
    wandb_project_name: str = field(default="sparse_generative_retrieval_tmp")
    pad_token_id: Optional[int] = field(default=None)
    smtid_as_docid: bool = field(default=False)

    # for evaluation
    eval_queries_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/raw.tsv")
    eval_qrels_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json")
    
    

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, self.run_name, "checkpoint")
        if self.local_rank <= 0:
            os.makedirs(self.output_dir, exist_ok=True)

        assert self.loss_type in {"margin_mse", "kldiv", "twins_margin_mse", "listwise_decode", "sprefix_nway_decode", 
                                  "pseudo_query_to_docid", "sparse_project_margin_mse", "sparse_project_margin_mse_dyn",
                                  "sparse_project_pretrain_margin_mse",
                                  "sparse_subset_k_project_margin_mse_dyn",
                                  "sparse_query_to_docid",
                                  "sparse_query_to_mul_docid",
                                  "t5seq_pretrain_margin_mse",
                                  "t5seq_query_to_docid",
                                  "t5seq_gumbel_max_margin_mse",
                                  "t5_aq_encoder_margin_mse",
                                  "t5seq_aq_encoder_margin_mse",
                                  "t5seq_aq_encoder_ranknet",
                                  "t5seq_aq_encoder_seq2seq",
                                  "t5seq_aq_encoder_knp_margin_mse",
                                  "t5seq_aq_encoder_lng_knp_margin_mse",
                                  "t5seq_aq_encoder_lng_knp_margin_mse_and_seq2seq",
                                  "t5seq_aq_encoder_decomp_margin_mse",
                                  "bert_bce",
                                  "t5seq_bce"}, self.loss_type

        if isinstance(self.task_names, list):
            pass 
        elif isinstance(self.task_names, str):
            self.task_names = ujson.loads(self.task_names)
        else:
            raise ValueError(f"task_names: {self.task_names} don't have valid type.")
        
        # mannualy set ln_to_weight 
        self.ln_to_weight = {}
        for ln in self.task_names:
            if ln == "rank":
                self.ln_to_weight[ln] = 1. 
            elif ln == "commit":
                self.ln_to_weight[ln] = 1. 
            elif ln == "reg":
                self.ln_to_weight[ln] = 1.0 
            elif ln == "early_rank":
                self.ln_to_weight[ln] = 1.
            elif ln == "rank_4":
                self.ln_to_weight[ln] = 1.0 
            elif ln == "rank_8":
                self.ln_to_weight[ln] = 1.0
            elif ln == "rank_16":
                self.ln_to_weight[ln] = 1.0
            elif ln == "rank_32":
                self.ln_to_weight[ln] = 1.0
            elif ln == "seq_4":
                self.ln_to_weight[ln] = 1.0
            elif ln == "seq_8":
                self.ln_to_weight[ln] = 1.0
            elif ln == "seq_16":
                self.ln_to_weight[ln] = 1.0
            elif ln == "seq":
                self.ln_to_weight[ln] = 1.0
            elif ln == "cls":
                self.ln_to_weight[ln] = 1.0
            else:
                print(type(self.task_names), self.task_names[0], self.task_names)
                raise ValueError(f"loss name: {ln} is not valid")
        print("task_names", self.task_names, self.ln_to_weight)
        

@dataclass
class EvalArguments:
    collection_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/full_collection")
    pretrained_path: str = field(default="")
    index_dir: str = field(default="")
    mmap_dir: str = field(default="")
    initial_mmap_dir: str = field(default="")
    target_mmap_dir: str = field(default="")
    out_dir: str = field(default="")
    flat_index_dir: str = field(default=None)
    model_name_or_path: str = field(default="t5-base")
    max_length: int = field(default=256)
    index_retrieve_batch_size: int = field(default=256)
    local_rank: int = field(default=-1)
    task: str = field(default="")
    topk: int = field(default=1000)
    q_collection_paths: List[str] = field(default_factory=lambda: ["/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/queries_2019/",
                          "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/queries_2020/",
                          "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/"])
    eval_qrel_path: List[str] = field(default_factory=lambda: [
        "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json",
        "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel.json",
        "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2019/qrel_binary.json",
        "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel.json",
        "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/TREC_DL_2020/qrel_binary.json"
    ])
    eval_metric: List[List[str]] = field(default_factory=lambda: [
        ["mrr_10", "recall"],
        ["ndcg_cut"],
        ["mrr_10", "recall"],
        ["ndcg_cut"],
        ["mrr_10", "recall" ]])
    docid_to_smtid_path: Optional[str] = field(default=None)
    initial_docid_to_smtid_path: Optional[str] = field(default=None)
    qid_to_smtid_path: Optional[str] = field(default=None)
    use_fp16: bool = field(default=False)
    num_return_sequences: int = field(default=10)
    encoder_type: str = field(default="standard_encoder")
    train_query_dir: Optional[str] = field(default=None)
    out_qid_to_smtid_dir: Optional[str] = field(default=None)
    centroid_path: Optional[str] = field(default=None)
    centroid_idx: Optional[int] = field(default=None)
    dev_queries_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/raw.tsv")
    dev_qrels_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json")
    retrieve_json_path: str = field(default=None)
    num_subvectors_for_pq: int = field(default=32)
    codebook_bits: int = field(default=8)
    eval_run_paths: List[str] = field(default_factory=lambda: {
        "MSMARCO":"/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/MSMARCO/run.json",
        "TREC_DL_2019": "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/TREC_DL_2019/run.json",
        "TREC_DL_2020": "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-t5seq-aq/t5_docid_gen_encoder_1/out/TREC_DL_2020/run.json"
    })
    #eval_run_paths: List[str] = field(default_factory=lambda: {
    #    "MSMARCO":"/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq//t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/out_docid_from_sub_4_top2000/MSMARCO/run.json",
    #    "TREC_DL_2019": "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/out_docid_from_sub_4_top2000/TREC_DL_2019/run.json",
    #    "TREC_DL_2020": "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-1M-t5seq-aq/t5seq_aq_encoder_seq2seq_1_lng_knp_self_mnt_32_dcy_2/out_docid_from_sub_4_top2000/TREC_DL_2020/run.json"
    #})
    batch_size: Optional[int] = field(default=64)
    topk: int = field(default=200)
    apply_log_softmax_for_scores: Optional[bool] = field(default=False)
    max_new_token: Optional[int] = field(default=None)
    max_new_token_for_docid: int = field(default=32)
    num_links: int = field(default=100)
    ef_construct: int = field(default=128)


    def __post_init__(self):
        assert self.encoder_type in ["standard_encoder", "sparse_project_encoder", "sparse_subset_k_project_encoder",
                                     "sparse_project_pretrain_encoder", "t5seq_pretrain_encoder", "t5seq_gumbel_max_encoder"]
    
@dataclass
class RerankArguments:
    collection_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/full_collection")
    out_dir: str = field(default="")
    model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_length: int = field(default=256)
    batch_size: int = field(default=64)
    #q_collection_paths: List[str] = field(default_factory=lambda: [
    #    "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/labeled_queries/"
    #])
    #run_json_paths: List[str] = field(default_factory=lambda: [
    #    "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments/t5model_encoder_marginmse_again/out/MSMARCO_TRAIN/run_with_qrel.json"
    #])
    q_collection_path: str = field(default="")
    run_json_path: str = field(default="")
    local_rank: int = field(default=-1)
    task: str = field(default=None)
    pseudo_queries_path: Optional[str] = field(default=None)
    docid_pseudo_qids_path: Optional[str] = field(default=None)
    json_type: str = field(default="jsonl")
    qid_smtid_docids_path: Optional[str] = field(default=None)

    # for sparse query_to_smtid evaluate 
    docid_to_smtid_path: str = field(default="")
    pretrained_path: str = field(default="")
    dev_queries_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_queries/raw.tsv")
    dev_qrels_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/dev_qrel.json")
    qid_docids_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/bm25_run/top1000.dev.rerank.json")
    query_to_smtid_tokenizer_type: str = field(default="t5-base")
    qid_smtid_rank_path: str = field(default="")
    train_qrels_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/qrels.json")
    train_queries_path: str = field(default="/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco/train_queries/queries/raw.tsv")
    qid_to_reldocid_hard_docids_path: Optional[str] = field(default=None)
    eval_qrel_path: Optional[str] = field(default=None)
    eval_metrics: Optional[str] = field(default=None)
@dataclass
class TreeConstructArguments:
    mmap_path: str = field(default="")
    out_dir: str = field(default="")
    balance_factor: int = field(default=10)
    leaf_factor: int = field(default=100)
    sampled_doc_num: int = field(default=100_000)
    kmeans_pkg: str = field(default="faiss")
    
