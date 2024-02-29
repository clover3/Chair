import torch

from .cross_encoder_trainer import CrossEncoderTransformerTrainer
from ..tasks import amp
from ..utils.utils import parse


def combine(out_left, out_right):
    d1 = parse("left", out_left)
    d2 = parse("right", out_right)

    out_d = {}
    for k, v1 in d1.items():
        v2 = d2[k]
        if k == "logits":
            out_d[k] = v1 + v2
        else:
            out_d[k] = v1, v2
    return out_d


class PEPTrainer(CrossEncoderTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch):
        # for this trainer, the batch contains query, pos doc and neg doc HF formatted inputs
        out_d = {}
        for doc_role in ["pos", "neg"]:
            args_per_qd = {"kwargs": parse(batch, doc_role)}
            out_per_qd = {}
            for seg_role in ["left", "right"]:
                model_args = {"kwargs": parse(args_per_qd, seg_role)}
                with torch.cuda.amp.autocast() if self.fp16 else amp.NullContextManager():
                    out = self.model(**model_args)
                    out_per_qd[seg_role] = out

            out_combined = combine(out_per_qd["left"], out_per_qd["right"])
            for k, v in out_combined.items():
                out_d[f"{doc_role}_{k}"] = v

        if "teacher_p_score" in batch:  # distillation pairs dataloader
            out["teacher_pos_score"] = batch["teacher_p_score"]
            out["teacher_neg_score"] = batch["teacher_n_score"]
        return out
