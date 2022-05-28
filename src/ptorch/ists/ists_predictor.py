import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from cpath import common_model_dir_root
from ptorch.ists.bert_chunk_encoder import BertChunkEncoderForInference
from ptorch.ists.holder import Holder
from ptorch.ists.pointer_network_model import PointerNetwork


def get_cfg():
    cfg = Holder()
    cfg.constr_res_path = ""
    cfg.rho = 4.0
    cfg.gpuid = 0
    cfg.input_dim = 1536
    cfg.output_constr = ''
    cfg.hidden_dim = 768
    return cfg


class ISTSPredictorI(ABC):
    @abstractmethod
    def predict(self, left_sent: List[str], right_sent: List[str]) -> np.array:
        pass


class ISTSPredictor(ISTSPredictorI):
    def __init__(self, model_path):
        cfg = get_cfg()
        pointer_network = PointerNetwork(cfg)
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torched_loaded = torch.load(model_path, map_location=map_location)
        pointer_network.load_state_dict(torched_loaded)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        self.encoder = BertChunkEncoderForInference(tokenizer, model)
        self.pointer_network = pointer_network

    def predict(self, left_sent: List[str], right_sent: List[str]) -> np.array:
        max_len = max([len(left_sent), len(right_sent)])
        left_embedding = self.encoder.encode(left_sent, max_len)
        right_embedding = self.encoder.encode(right_sent, max_len)

        inputs = {
            "left_embedding": left_embedding,
            "right_embedding": right_embedding,
            "num_left_chunks": len(left_sent),
            "num_right_chunks": len(right_sent),
        }
        outputs = self.pointer_network.forward(inputs)
        log_prob_matrix, g_values = outputs
        prob_matrix = torch.exp(log_prob_matrix)
        return prob_matrix.detach().numpy()


def get_ists_predictor() -> ISTSPredictor:
    checkpoint_path = os.path.join(common_model_dir_root, "ists", "pointer_network", "model.checkpoint")
    predictor = ISTSPredictor(checkpoint_path)
    return predictor
