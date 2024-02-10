import torch
from abc import ABC

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class CrossEncoderLikeBase(torch.nn.Module, ABC):
    def __init__(self, model_type_or_dir, **kwargs):
        super().__init__()
        self.bert_like = AutoModelForSequenceClassification.from_pretrained(
            model_type_or_dir, num_labels = 1)

    def forward(self, **kwargs):
        out = {}
        output: SequenceClassifierOutput = self.bert_like(**kwargs["kwargs"])
        score: torch.FloatTensor = output.logits
        out.update({"score": score})
        return out
