import torch
import torch.nn as nn 
from transformers import AutoModel 

class DenseEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
    def query_encode(self, **inputs):
        rep = self.model(**inputs).last_hidden_state[:,0,:]
        return rep