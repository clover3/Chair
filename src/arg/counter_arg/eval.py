import enum
from collections import Iterable
from typing import Callable
from typing import List

from arg.counter_arg import es
from arg.counter_arg.data_loader import load_labeled_data
from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import ArguDataPoint
from arg.counter_arg.header import Passage, ArguDataID
from distrib.parallel import parallel_run
from list_lib import lmap, max_idx
from misc_lib import average


class EvalCondition(enum.Enum):
    SameDebateArguments = 1
    SameThemeCounters = 2
    SameThemeArguments = 3
    EntirePortalCounters = 4
    EntirePortalArguments = 5


def get_eval_payload(split) -> List[Passage]:
    itr: Iterable[ArguDataPoint] = load_labeled_data(split)

    r = []
    for item in itr:
        r.append(item.text1)
    return r


def get_eval_payload_from_dp(itr) -> List[Passage]:
    r = []
    for item in itr:
        r.append(item.text1)
    return r


def eval_correctness(predictions: List[ArguDataID], gold_labels: List[ArguDataPoint]) -> List[bool]:
    assert len(predictions) == len(gold_labels)

    correctness: List[bool] = []
    for pred, gold in zip(predictions, gold_labels):
        is_correct = pred.id == gold.text2.id.id
        correctness.append(is_correct)

    return correctness


def retrieve_candidate(p: Passage, split, condition: EvalCondition) -> List[ArguDataID]:
    r = []

    def is_valid_candidate(source_id, target_id):
        if source_id == target_id:
            return False

        split, theme, debate_name, stance, sub_name = source_id.split("/")
        split2, theme2, debate_name2, stance2, sub_name2 = target_id.split("/")
        if condition == EvalCondition.SameDebateArguments:
            if debate_name != debate_name2:
                return False
            if theme != theme2:
                return False
        elif condition == EvalCondition.SameThemeArguments:
            if theme != theme2:
                return False
        elif condition == EvalCondition.SameThemeCounters:
            if theme != theme2:
                return False
            if "counter" not in sub_name2:
                return False
        elif condition == EvalCondition.EntirePortalCounters:
            if "counter" not in sub_name2:
                return False
        return True
    for text, data_id, score in es.search(split, p.text, 100):
        if is_valid_candidate(p.id.id, data_id):
            r.append(ArguDataID(id=data_id))
    return r


def run_eval(split,
            scorer: Callable[[Passage, List[Passage]], float],
             condition: EvalCondition):
    all_candidate = list(enum_all_argument(split))
    candidate_d = {c.id: c for c in all_candidate}
    problems: List[ArguDataPoint] = list(load_labeled_data(split))
    problems = problems[:30]
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    correctness = []
    for query, problem in zip(payload, problems):
        p = problem
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate = list([candidate_d[x] for x in candidate_ids])
        scores = scorer(query, candidate)
        best_idx = max_idx(scores)
        pred_item: Passage = candidate[best_idx]
        gold_id = p.text2.id
        pred_id = pred_item.id
        correct = gold_id == pred_id
        content_equal = (p.text2.text == pred_item.text)
        correct = correct or content_equal
        #
        print("-------------------", correct, content_equal)
        print("query:", p.text1.id)
        # print(p.text1.text)
        print("gold:", p.text2.id)
        # print(p.text2.text)
        print("pred:", pred_item.id)
        # print(scores[best_idx].name)
        # print(pred_item.text)

        correctness.append(correct)
    avg_p_at_1 = average(correctness)
    return avg_p_at_1


def eval_thread(param):
    payload: List[Passage] = param[0]
    split, predictor_getter = param[1]
    predictor = predictor_getter(split)
    all_candidate = list(enum_all_argument(split))
    candidate_d = {c.id: c for c in all_candidate}

    def pred_inst(query_p: Passage) -> Passage:
        candidate_ids: List[ArguDataID] = retrieve_candidate(query_p, split)
        candidate = list([candidate_d[x] for x in candidate_ids])
        pred = predictor(query_p, candidate)
        return pred
    return lmap(pred_inst, payload)



def run_eval_threaded(split, predictor_getter):
    print("Loading data..")
    problems: List[ArguDataPoint] = list(load_labeled_data(split))
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    print("starting predictions")
    predictions = parallel_run(payload, (split, predictor_getter), eval_thread, 5)
    correctness = eval_correctness(predictions, problems)
    avg_p_at_1 = average(correctness)
    print(avg_p_at_1)


