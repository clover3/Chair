from abc import ABC

import torch
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn


class CrossEncoderLikeBase(torch.nn.Module, ABC):
    def __init__(self, model_type_or_dir, **kwargs):
        super().__init__()
        self.bert_like = AutoModelForSequenceClassification.from_pretrained(
            model_type_or_dir, num_labels=1)

    def forward(self, **kwargs):
        out = {}
        output: SequenceClassifierOutput = self.bert_like(**kwargs["kwargs"])
        score: torch.FloatTensor = output.logits
        out.update({"score": score})
        return out


# For each pass
class TransformerWeightedSum(torch.nn.Module, ABC):
    def __init__(self, model_type_or_dir, **kwargs):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(
            model_type_or_dir, num_labels=1)
        hidden_size = self.transformer.config.hidden_size
        self.weight_predictor = nn.Linear(hidden_size, 1)
        self.score_predictor = nn.Linear(hidden_size, 1)

    def forward(self, **kwargs):
        kwargs = kwargs["kwargs"]
        kwargs_b = {k: v for k, v in kwargs.items() if k != "doc_cls_indices"}
        output = self.transformer(output_hidden_states=True, **kwargs_b)

        feature_rep = output.last_hidden_state
        doc_cls_indices = kwargs["doc_cls_indices"]  # [B, L, ]
        raw_weights = self.weight_predictor(feature_rep)
        mask = 1 - doc_cls_indices
        raw_weights = raw_weights + torch.unsqueeze(mask * -1e10, -1)
        scores = self.score_predictor(feature_rep)
        weights = nn.functional.softmax(raw_weights, dim=1)
        score = torch.sum(scores * weights, dim=1)
        out = {
            "score": score,
            "weights": weights,
            "scores": scores,
            "doc_cls_indices": doc_cls_indices
        }
        return out
