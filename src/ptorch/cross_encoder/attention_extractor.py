import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from list_lib import left, right
from misc_lib import batch_iter_from_entry_iter
from trainer_v2.chair_logging import c_log


class AttentionExtractor:
    def __init__(self, inf_batch_size=100):
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.inf_batch_size = inf_batch_size

    def predict_inner(self, qd_list):
        c_log.debug("predict_inner(%d items)", len(qd_list))
        q_list = left(qd_list)
        d_list = right(qd_list)
        features = self.tokenizer(q_list, d_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features.to(self.device)
            output = self.model(**features, output_attentions=True)
            scores = np.array(output.logits.cpu())

            attn = torch.stack(output.attentions, dim=1)
            attn = torch.mean(attn, dim=1)
            attn = torch.mean(attn, dim=1)
            attn = np.array(attn.cpu())
            outputs = []

            for idx in range(len(q_list)):
                out_row = q_list[idx], d_list[idx], scores[idx, 0], attn[idx]
                outputs.append(out_row)

            return outputs

    def predict(self, qd_list):
        return self.predict_itr(qd_list)

    def predict_itr(self, qd_list):
        for sub_batch in batch_iter_from_entry_iter(qd_list, self.inf_batch_size):
            items: List[Tuple[str, str, float, np.array]] = self.predict_inner(sub_batch)
            yield from items