from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy
from list_lib import right, left
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

def get_ce_msmarco_mini_lm_score_fn():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.eval()

    def score_fn(qd_list: List[Tuple[str, str]]) -> List[float]:
        q_list = left(qd_list)
        d_list = right(qd_list)
        features = tokenizer(q_list, d_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features.to(device)
            scores = model(**features).logits.cpu()
        return numpy.array(scores[:, 0]).tolist()

    return score_fn
