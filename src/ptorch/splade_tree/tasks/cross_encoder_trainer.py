import torch

from ptorch.splade_tree.tasks.transformer_trainer import TransformerTrainer
from ..tasks import amp
from ..tasks.base.trainer import TrainerIter
from ..utils.metrics import init_eval
from ..utils.utils import parse
import json
import os
from collections import defaultdict


class CrossEncoderTransformerTrainer(TransformerTrainer):
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

    def evaluate_loss(self, data_loader):
        """loss evaluation
        """
        out_d = defaultdict(float)
        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            out = self.forward(batch)
            val_ranking_loss = self.loss(out).mean().item()
            out_d["val_ranking_loss"] += val_ranking_loss
            if self.regularizer is not None:
                if "train" in self.regularizer:
                    total_loss = val_ranking_loss
                    for reg in self.regularizer["train"]:
                        lambda_q = self.regularizer["train"][reg]["lambdas"]["lambda_q"].get_lambda() if "lambda_q" in \
                                                                                                         self.regularizer[
                                                                                                             "train"][
                                                                                                             reg][
                                                                                                             "lambdas"] else False
                        lambda_d = self.regularizer["train"][reg]["lambdas"]["lambda_d"].get_lambda() if "lambda_d" in \
                                                                                                         self.regularizer[
                                                                                                             "train"][
                                                                                                             reg][
                                                                                                             "lambdas"] else False
                        targeted_rep = self.regularizer["train"][reg]["targeted_rep"]
                        r_loss = 0
                        if lambda_q:
                            r_loss += (self.regularizer["train"][reg]["loss"](
                                out["pos_q_{}".format(targeted_rep)]) * lambda_q).mean().item()

                        if lambda_d:
                            r_loss += ((self.regularizer["train"][reg]["loss"](
                                out["pos_d_{}".format(targeted_rep)]) * lambda_d).mean().item() + (
                                               self.regularizer["train"][reg]["loss"](
                                                   out["neg_d_{}".format(targeted_rep)]) * lambda_d).mean().item()) / 2
                        out_d["val_{}_loss".format(reg)] += r_loss
                        total_loss += r_loss
                    out_d["val_total_loss"] += total_loss
        return {key: value / len(data_loader) for key, value in out_d.items()}

    def evaluate_full_ranking(self, i):
        raise NotImplementedError

    def save_checkpoint(self, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model  # when using DataParallel
        # it is practical (although redundant) to save model weights using huggingface API, because if the model has
        # no other params, we can reload it easily with .from_pretrained()
        output_dir = os.path.join(self.config["checkpoint_dir"], "model")
        model_to_save.transformer_rep.transformer.save_pretrained(output_dir)
        tokenizer = model_to_save.transformer_rep.tokenizer
        tokenizer.save_pretrained(output_dir)
        if model_to_save.transformer_rep_q is not None:
            output_dir_q = os.path.join(self.config["checkpoint_dir"], "model_q")
            model_to_save.transformer_rep_q.transformer.save_pretrained(output_dir_q)
            tokenizer = model_to_save.transformer_rep_q.tokenizer
            tokenizer.save_pretrained(output_dir_q)
        super().save_checkpoint(**kwargs)
