from typing import Tuple, List

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import token_list_to_segmented_text
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from contradiction.medical_claims.token_tagging.batch_solver_common import BSAdapterIF, BatchSolver
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import left, right
from misc_lib import average, tprint

ESig = Tuple[NLIInput, Tuple[int, int]]
ESig2 = Tuple[Tuple[str, str], Tuple[int, int]]


def overlap_window_attribution(l2: int, output_):
    scores_building = [list() for _ in range(l2)]
    for score, (st, ed) in output_:
        for i in range(st, ed):
            scores_building[i].append(score)
    ret = [average(l) for l in scores_building]
    return ret


class PartialSegSolvingAdapter(BSAdapterIF):
    def __init__(self, predict_fn, sel_score_fn):
        self.tokenizer = get_tokenizer()
        self.predict_fn = predict_fn
        self.sel_score_fn = sel_score_fn


    def neural_worker(self, items: List[ESig]):
        indice_list: List[Tuple[int, int]] = right(items)
        nli_input_list: List[NLIInput] = left(items)
        tprint("Sending {} items".format(len(items)))
        probs_list: List[List[float]] = self.predict_fn(nli_input_list)
        tprint("Done")
        return list(zip(probs_list, indice_list))

    def reduce(self, t1: List[str], t2: List[str],
               output: List[Tuple[List[float], Tuple[int, int]]]) -> List[float]:

        l2 = len(t2)
        output_t = [(self.sel_score_fn(probs), (st, ed)) for probs, (st, ed) in output]
        ret = overlap_window_attribution(l2, output_t)
        return ret

    def enum_child(self, text1_tokens, text2_tokens) -> List[ESig]:
        t1: SegmentedText = token_list_to_segmented_text(self.tokenizer, text1_tokens)
        t2: SegmentedText = token_list_to_segmented_text(self.tokenizer, text2_tokens)
        es_list = []
        for window_size in [1, 3, 6]:
            for offset in range(0, min(window_size, 3)):
                st = offset
                while st < t2.get_seg_len():
                    ed = min(st + window_size, t2.get_seg_len())
                    t2_sub = t2.get_sliced_text(list(range(st, ed)))
                    es = (NLIInput(t1, t2_sub), (st, ed))
                    es_list.append(es)
                    st = st + window_size
        return es_list


def get_batch_partial_seg_solver(sel_score_fn) -> BatchSolver:
    predict_fn = get_nli_client("localhost")
    adapter = PartialSegSolvingAdapter(predict_fn, sel_score_fn)
    return BatchSolver(adapter)



class PartialSegSolvingAdapter2(BSAdapterIF):
    def __init__(self, predict_fn, sel_score_fn):
        self.tokenizer = get_tokenizer()
        self.predict_fn = predict_fn
        self.sel_score_fn = sel_score_fn

    def neural_worker(self, items: List[ESig2]):
        indice_list: List[Tuple[int, int]] = right(items)
        nli_input_list: List[Tuple[str, str]] = left(items)
        tprint("Sending {} items".format(len(items)))
        probs_list: List[List[float]] = self.predict_fn(nli_input_list)
        tprint("Done")
        return list(zip(probs_list, indice_list))

    def reduce(self, t1: List[str], t2: List[str],
               output: List[Tuple[List[float], Tuple[int, int]]]) -> List[float]:

        l2 = len(t2)
        output_t = [(self.sel_score_fn(probs), (st, ed)) for probs, (st, ed) in output]
        ret = overlap_window_attribution(l2, output_t)
        return ret

    def enum_child(self, text1_tokens, text2_tokens) -> List[ESig2]:
        es_list = []
        text1 = " ".join(text1_tokens)

        l2 = len(text2_tokens)
        for window_size in [1, 3, 6]:
            for offset in range(0, min(window_size, 3)):
                st = offset
                while st < l2:
                    ed = min(st + window_size, l2)
                    t2_sub = " ".join(text2_tokens[st:ed])
                    es = (text1, t2_sub), (st, ed)
                    es_list.append(es)
                    st = st + window_size
        return es_list