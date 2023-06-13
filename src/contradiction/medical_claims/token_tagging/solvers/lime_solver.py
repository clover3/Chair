from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF, \
    TokenScoringSolverIFOneWay
from list_lib import right, left
from misc_lib import get_first
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict_empty
from trainer_v2.keras_server.bert_like_client import BERTClientCore
from trainer_v2.keras_server.bert_like_server import get_keras_bert_like_predict_fn
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_client, get_nli14_cache_client, get_nli14_direct, \
    get_cached_client
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np

from trainer_v2.train_util.get_tpu_strategy import get_strategy


def split(s):
    return s.split()



from lime import lime_text


class LimeSolver(TokenScoringSolverIFOneWay):
    def __init__(self, predict_fn, target_label):
        self.predict_fn = predict_fn
        self.target_label = target_label

    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        scores1 = self.solve_one_way(text2_tokens, text1_tokens)
        scores2 = self.solve_one_way(text1_tokens, text2_tokens)
        return scores1, scores2

    def solve_one_way(self, text1_tokens: List[str], text2_tokens: List[str]):
        text1 = " ".join(text1_tokens)
        entry = " ".join(text2_tokens)
        explainer = lime_text.LimeTextExplainer(
            split_expression=split, bow=False, random_state=0
        )
        def classifier_fn(text_list):
            payload = [(text1, text) for text in text_list]
            return np.array(self.predict_fn(payload))

        len_seq = len(text2_tokens)
        target_label = self.target_label
        exp = explainer.explain_instance(entry, classifier_fn, labels=(target_label,),
                                         num_features=len_seq, num_samples=500)
        scored_items = list(exp.local_exp[target_label])
        scored_items.sort(key=get_first)
        indices = left(scored_items)
        for i in range(len(indices)):
            if not indices[i] == i:
                print(indices)
                raise Exception
        scores = right(scored_items)
        return scores


def get_lime_solver_nli14_direct(target_idx):
    predict_fn = get_nli14_direct(get_strategy())
    solver = LimeSolver(predict_fn, target_idx)
    return solver
