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
                        default=None,
                        type=str)
    parser.add_argument("--K",
                        default=256,
                        type=int)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    model_dir = args.model_dir
    K = args.K
    print("model_dir: ", model_dir, "K: ", K)

    d_model = 768
    if K != 256:
        assert str(K) in model_dir
    else:
        print("using default value for K: ", K)

    pretrained_path = os.path.join(model_dir, "checkpoint")
    shared_output_input_embeds = False
    if not shared_output_input_embeds:
        out_dir = os.path.join(model_dir, "no_share_checkpoint")
    else:
        out_dir = os.path.join(model_dir, "share_checkpoint")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    index_path = os.path.join(model_dir, "aq_index/model.index")

    index = faiss.read_index(index_path)
    rq = index.rq

    print("M, K, d_model are: ", rq.M, K, d_model)

    centroids = faiss.vector_to_array(rq.codebooks).reshape(rq.M, K, 768)


    model = T5SeqAQEncoder.from_pretrained(pretrained_path, shared_output_input_embeds=shared_output_input_embeds)

    print("before modifying embed_layers")
    for name, param in model.base_model.named_parameters():
        if "list_decoder_embeds" in name or "list_output_embeds" in name:
            print(name, param.shape)

    # change embed layers and corresponding config params
    if shared_output_input_embeds:
        model.base_model.list_decoder_embeds = nn.ModuleList([
                nn.Embedding(K, d_model) for _ in range(rq.M)
            ])
    else:
        model.base_model.list_decoder_embeds = nn.ModuleList([
                nn.Embedding(K, d_model) for _ in range(rq.M)
            ])
        model.base_model.list_output_embeds = nn.ModuleList([
                nn.Embedding(K, d_model) for _ in range(rq.M)
            ])
    model.base_model.config.decoder_vocab_sizes = [K for _ in range(rq.M)]
    model.config.decoder_vocab_sizes = [K for _ in range(rq.M)]
    model.base_model.config.max_decoder_length = rq.M
    model.config.max_decoder_length = rq.M
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
