import torch

from ptorch.splade_tree.tasks.transformer_trainer import TransformerTrainer
from .transformer_trainer2 import TransformerTrainer2
from ..tasks import amp
from ..tasks.base.trainer import TrainerIter
from ..utils.metrics import init_eval
from ..utils.utils import parse
import json
import os
from collections import defaultdict


class CrossEncoderTransformerTrainer(TransformerTrainer2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        # for this trainer, the batch contains query, pos doc and neg doc HF formatted inputs
        pos_args = {"kwargs": parse(batch, "pos")}
        neg_args = {"kwargs": parse(batch, "neg")}
        with torch.cuda.amp.autocast() if self.fp16 else amp.NullContextManager():
            out_pos = self.model(**pos_args)
            out_neg = self.model(**neg_args)
        out = {}
        for k, v in out_pos.items():
            out["pos_{}".format(k)] = v
        for k, v in out_neg.items():
            out["neg_{}".format(k)] = v
        if "teacher_p_score" in batch:  # distillation pairs dataloader
            out["teacher_pos_score"] = batch["teacher_p_score"]
            out["teacher_neg_score"] = batch["teacher_n_score"]
        return out
