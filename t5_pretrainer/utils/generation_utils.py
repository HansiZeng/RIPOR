from typing import List, Dict
import math
import os 
import ujson

import torch
from transformers.generation_logits_process import LogitsProcessor

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)
    
class PrefixConstrainedLogitsProcessorForSmtidTree(LogitsProcessor):
    def __init__(self, list_smtids, eos_token_id):
        self.trie = Trie(list_smtids)
        self.eos_token_id = eos_token_id

    def _prefix_allowed_tokens_fn(self, sent):
        """
        Args:
            sent: List[int]
        Returns:
            allowed_tokens: List[int]
        """
        if isinstance(sent, torch.Tensor):
            sent = sent.tolist()
        elif isinstance(sent, list):
            pass 
        else:
            raise ValueError(f"{sent} don't have valid type.")
        
        return self.trie[sent]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, sent in enumerate(input_ids):
            allowed_tokens = self._prefix_allowed_tokens_fn(sent)

            # if allowed_tokens is empty, we end the sent
            if len(allowed_tokens) == 0:
                allowed_tokens.append(self.eos_token_id)

            mask[batch_id, allowed_tokens] = 0

        return scores + mask
    

if __name__ == "__main__":
    from ..dataset.dataset import CollectionDatasetWithDocIDPreLoad
    from ..dataset.dataloader import CollectionDataWithDocIDLoader
    from ..modeling.t5_generative_retriever import T5SeqQueryToDocidEncoder
    from transformers.generation_logits_process import LogitsProcessorList
    from tqdm import tqdm

    root_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/t5_pretrainer/t5_pretrainer"
    docid_to_smtid_path = os.path.join(root_dir, "experiments-1Mtree-t5seq-encoder-1000/t5_docid_gen_encoder_2_0/btree_1000/docid_to_smtid.json")
    pretrained_path = os.path.join(root_dir, "experiments-1Mtree-t5seq-encoder-1000/t5_docid_gen_encoder_2_1/checkpoint")
    dev_query_dir = "/home/ec2-user/quic-efs/user/hansizeng/work/data/msmarco-1M/dev_queries/"

    with open(docid_to_smtid_path) as fin:
        docid_to_smtids = ujson.load(fin)

    # create model and dataset 
    model = T5SeqQueryToDocidEncoder.from_pretrained(pretrained_path)
    model = model.base_model 
    model.config.decoding = True 
    model.cuda() 

    dev_dataset = CollectionDatasetWithDocIDPreLoad(data_dir=dev_query_dir, id_style="row_id",
                                                tid_to_smtid_path=None, add_prefix=True, is_query=True)
    dev_loader =  CollectionDataWithDocIDLoader(dataset=dev_dataset, 
                                                tokenizer_type=pretrained_path,
                                                max_length=64,
                                                batch_size=4,
                                                num_workers=4)

    # parameters for generate
    pad_token_id = 1001
    eos_token_id = 1000
    decoder_start_token_id = -1
    num_beams = 100
    num_return_sequences = 100
    assert pad_token_id ==  model.config.decoder_vocab_sizes[0] - 1, (pad_token_id,  model.config.decoder_vocab_sizes[0])

    all_smtids = [list(xs) for xs in docid_to_smtids.values()]
    logits_processor = LogitsProcessorList([PrefixConstrainedLogitsProcessorForSmtidTree(all_smtids, eos_token_id=eos_token_id)])

    smtid_lengths = [len(xs) for xs in all_smtids]
    max_new_tokens = max(smtid_lengths) - 2  # -1 and eos_token_id is added 
    print("max_new_tokens", max_new_tokens)

    # start to decoding 
    all_predicted_smtids = []
    all_docids = []
    for i, batch in enumerate(tqdm(dev_loader,total=len(dev_loader))):
        with torch.no_grad():
            inputs = {k:v.cuda() for k, v in batch.items() if k != "id"}
            outputs = model.generate(inputs["input_ids"],
                           max_new_tokens=max_new_tokens,
                           pad_token_id=pad_token_id,
                           eos_token_id=eos_token_id,
                           decoder_start_token_id=decoder_start_token_id,
                           num_beams=num_beams,
                           num_return_sequences=num_return_sequences,
                           logits_processor=logits_processor,
                           output_scores=True,
                           return_dict=True,
                           return_dict_in_generate=True,)

        predicted_smtids = outputs.sequences.view(-1, num_beams, max_new_tokens+1).tolist()

        all_predicted_smtids.extend(predicted_smtids)
        all_docids.extend(batch["id"].cpu().tolist())

    
    #for docid, predicted_smtids in zip(all_docids, all_predicted_smtids):
       # print(docid, " || ", predicted_smtids)


    

