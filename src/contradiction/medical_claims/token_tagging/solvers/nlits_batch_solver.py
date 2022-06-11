import os
from typing import List, Iterable, Callable, Tuple

from contradiction.medical_claims.token_tagging.batch_solver_common import NeuralOutput, AdapterIF, BatchSolver
from cpath import common_model_dir_root
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten
from misc_lib import Averager
from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.custom_loop.demo.demo_common import enum_hypo_token_tuple_from_tokens, EncodedSegmentIF
from trainer_v2.custom_loop.demo.demo_two_seg_concat import get_two_seg_concat_encoder
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_ld_core(model_name):
    machine_name = get_local_machine_name()
    is_tpu = machine_name not in ["GOSFORD", "ingham.cs.umass.edu"]
    if is_tpu:
        model_path = f"gs://clovertpu/training/model/{model_name}/model_25000"
        model_path = f'/home/youngwookim/code/Chair/model/{model_name}/model_25000'
        strategy = get_strategy(True, "local")
    else:
        model_path = os.path.join(common_model_dir_root, 'runs', f"{model_name}", "model_25000")
        strategy = get_strategy(False, "")
    with strategy.scope():
        ld_core = LocalDecisionNLICore(model_path, strategy, 2)
    return ld_core


class ESTwoPiece(EncodedSegmentIF):
    def __init__(self, input_x,
                 t1,
                 t2,
                 h_tokens_list,
                 st, ed):
        self.input_x = input_x
        self.t1 = t1
        self.t2 = t2
        self.h_tokens_list = h_tokens_list
        self.st = st
        self.ed = ed

    def get_input(self):
        return self.input_x


class ScoreReducer:
    def __init__(self, target_label: int):
        self.target_label = target_label

    def reduce(self, records: List[Tuple[List[float], ESTwoPiece]], text2_tokens) -> List[float]:
        scores_building = [Averager() for _ in text2_tokens]
        for probs, es_item in records:
            s = probs[self.target_label]
            for i in range(es_item.st, es_item.ed):
                if i < len(scores_building):
                    scores_building[i].append(s)
        scores = [s.get_average() for s in scores_building]
        return scores


class NLITS40Adapter(AdapterIF):
    def __init__(self, nlits, target_label):
        self.nlits: LocalDecisionNLICore = nlits
        self.tokenizer = get_tokenizer()
        EncoderType = Callable[[List, List, List], Iterable[Tuple]]
        self.encode_two_seg_input: EncoderType = get_two_seg_concat_encoder()
        self.score_reducer = ScoreReducer(target_label)

    def enum_child(self, t1, t2):
        p_tokens = list(flatten(map(self.tokenizer.tokenize, t1)))
        es_list = []
        for h_first, h_second, st, ed in enum_hypo_token_tuple_from_tokens(self.tokenizer, t2, 3):
            x = self.encode_two_seg_input(p_tokens, h_first, h_second)
            es = ESTwoPiece(x, t1, t2, [h_first, h_second], st, ed)
            es_list.append(es)
        return es_list

    def neural_worker(self, items: List[ESTwoPiece]) -> List[Tuple[NeuralOutput, ESTwoPiece]]:
        l_decisions = self.nlits.predict_es(items)
        return list(zip(l_decisions, items))

    def reduce(self, t1, t2, item: List[Tuple[NeuralOutput, ESTwoPiece]]) -> List[float]:
        return self.score_reducer.reduce(item, t2)


def get_batch_solver_nlits40(target_label):
    nlits: LocalDecisionNLICore = get_ld_core("nli_ts_run40_0")
    adapter = NLITS40Adapter(nlits, target_label)
    solver = BatchSolver(adapter)
    return solver