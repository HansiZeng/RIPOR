import torch 
from ..modeling.t5_generative_retriever import T5SeqAQEncoder
import torch.nn as nn
import faiss 
import os
from transformers import AutoTokenizer
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", 
                        default="/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer/experiments-full-4-4096-28-256-t5seq-aq/t5_docid_gen_encoder_1/",
                        type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_dir = args.model_dir
    initial_K, target_K = 4096, 256
    print("model_dir: ", model_dir, "initial_K: ", initial_K, "target_K: ", target_K)

    d_model = 768

    pretrained_path = os.path.join(model_dir, "initial_no_share_checkpoint")
    shared_output_input_embeds = False
    if not shared_output_input_embeds:
        out_dir = os.path.join(model_dir, "no_share_checkpoint")
    else:
        out_dir = os.path.join(model_dir, "share_checkpoint")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    centroids = []
    data_pairs = [(os.path.join(model_dir, "initial_aq_index/model.index"), initial_K),
                  (os.path.join(model_dir, "target_aq_index/model.index"), target_K)]
    list_Ms = []
    list_Ks = []
    for i, (index_path, K) in enumerate(data_pairs):
        index = faiss.read_index(index_path)
        rq = index.rq
        arr_centroids = faiss.vector_to_array(rq.codebooks).reshape(rq.M, K, 768)
        print(f"index_path: {index_path}")
        print(f"centroid combination for level {i}: ",rq.M, K)
        centroids += [arr_centroids[i] for i in range(rq.M)]
        list_Ms.append(rq.M)
        list_Ks.append(K)

    print(f"length of centroids: {len(centroids)}, list_Ms: {list_Ms}, list_Ks: {list_Ks}")

    model = T5SeqAQEncoder.from_pretrained(pretrained_path, shared_output_input_embeds=shared_output_input_embeds)

    print("before modifying embed_layers")
    for name, param in model.base_model.named_parameters():
        if "list_decoder_embeds" in name or "list_output_embeds" in name:
            print(name, param.shape)

    # change embed layers and corresponding config params
    embedding_list = []
    shape_list = []
    max_M = 0
    for (M,K) in zip(list_Ms, list_Ks):
        embedding_list += [nn.Embedding(K, d_model) for _ in range(M)]
        shape_list += [K for _ in range(M)]
        max_M += M
    
    if shared_output_input_embeds:
        model.base_model.list_decoder_embeds = nn.ModuleList(embedding_list)
    else:
        model.base_model.list_decoder_embeds = nn.ModuleList(embedding_list)
        model.base_model.list_output_embeds = nn.ModuleList(embedding_list)
    model.base_model.config.decoder_vocab_sizes = shape_list
    model.config.decoder_vocab_sizes = shape_list
    model.base_model.config.max_decoder_length = max_M
    model.config.max_decoder_length = max_M
    
    print("after moddify embed_layers")
    for name, param in model.base_model.named_parameters():
        if "list_decoder_embeds" in name or "list_output_embeds" in name:
            print(name, param.shape)

    # assign the centroid embed for it.
    model.assign_output_embeds(centroids)

    # save model 
    model.save_pretrained(out_dir)

    # save tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.save_pretrained(out_dir)
