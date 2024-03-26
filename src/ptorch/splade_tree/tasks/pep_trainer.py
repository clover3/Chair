import torch

from .cross_encoder_trainer import CrossEncoderTransformerTrainer
from ..c2_log import c2_log
from ..tasks import amp
from ..utils.utils import parse


def combine(out_left, out_right):
    out_d = {}
    for k, v1 in out_left.items():
        v2 = out_right[k]
        if k == "score":
            out_d[k] = v1 + v2
    return out_d


class PEPTrainer(CrossEncoderTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        # for this trainer, the batch contains query, pos doc and neg doc HF formatted inputs
        out_d = {}
        for doc_role in ["pos", "neg"]:
            args_per_qd = parse(batch, doc_role)
            out_per_qd = {}
            for seg_role in ["left", "right"]:
                model_args = {"kwargs": parse(args_per_qd, seg_role)}
                with torch.cuda.amp.autocast() if self.fp16 else amp.NullContextManager():
                    out = self.model(**model_args)
                    out_per_qd[seg_role] = out

            out_combined = {}
            out_combined["score"] = out_per_qd["left"]["score"] + out_per_qd["right"]["score"]
            for k, v in out_combined.items():
                out_d[f"{doc_role}_{k}"] = v
        if "teacher_pos_score" in batch:  # distillation pairs dataloader
            out_d["teacher_pos_score"] = batch["teacher_pos_score"]
            out_d["teacher_neg_score"] = batch["teacher_neg_score"]
        return out_d
