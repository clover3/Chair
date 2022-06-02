import enum
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, NamedTuple

from arg.counter_arg import es
from arg.counter_arg.data_loader import load_labeled_data
from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import ArguDataPoint
from arg.counter_arg.header import Passage, ArguDataID
from cache import load_cache, save_to_pickle
from list_lib import lmap, get_max_idx
from misc_lib import average, NamedNumber
from utils.parallel import parallel_run


class EvalCondition(enum.Enum):
    SameDebateArguments = 1
    SameThemeCounters = 2
    SameThemeArguments = 3
    EntirePortalCounters = 4
    EntirePortalArguments = 5
    SameDebateClassification = 6


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


class ParsedID(NamedTuple):
    split: str
    theme: str
    debate_name: str
    stance: str
    sub_name: str

    @classmethod
    def from_str(cls, source_id:ArguDataID):
        split, theme, debate_name, stance, sub_name = source_id.id.split("/")
        return ParsedID(split, theme, debate_name, stance, sub_name)


def retrieve_candidate(p: Passage, split, condition: EvalCondition, top_k=100) -> List[ArguDataID]:
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
        elif condition == EvalCondition.SameDebateClassification:
            return True
        return True

    text = p.text
    max_len = 3000
    q_text = text[:max_len]
    for text, data_id, score in es.search(split, q_text, top_k):
        if is_valid_candidate(p.id.id, data_id):
            r.append(ArguDataID(id=data_id))
    return r


def prepare_eval_data(split) -> Tuple[List[ArguDataPoint], Dict[ArguDataID, Passage]]:
    problems = load_problems(split)
    candidate_d = load_candidate_d(split)

    return problems, candidate_d


def load_candidate_d(split):
    candidate_cache_name = "argu_candidate_{}".format(split)
    candidate_d: Dict[ArguDataID, Passage] = load_cache(candidate_cache_name)
    if candidate_d is None:
        all_candidate: Iterable[Passage] = list(enum_all_argument(split))
        candidate_d = {c.id: c for c in all_candidate}
        save_to_pickle(candidate_d, candidate_cache_name)
    else:
        print("using cache for candidate")
    return candidate_d


def load_problems(split) -> List[ArguDataPoint]:
    problem_cache_name = "argu_problem_{}".format(split)
    problems: List[ArguDataPoint] = load_cache(problem_cache_name)
    if problems is None:
        problems: List[ArguDataPoint] = list(load_labeled_data(split))
        save_to_pickle(problems, problem_cache_name)
    else:
        print("Using cache for problems")
    return problems


# 0 : success
# 1 : Same topic, is counter, valid stance, but not direct one
# 2 : Same topic, is counter, wrong stance
# 3 : Same topic, is point, stance is valid
# 4 : Same topic, is point, stance is not valid
# 5 : Same topic, not opposing stance
# 6 : not same topic

def failure_type(gold_id, pred_id):
    if gold_id == pred_id:
        return 0

    tokens1 = gold_id.split("/")
    tokens2 = pred_id.split("/")

    if "/".join(tokens1[:-2]) == "/".join(tokens2[:-2]):
        stance1 = tokens1[-2]
        stance2 = tokens2[-2]

        is_counter = "counter" in tokens2[-1]
        if is_counter:
            if stance1 == stance2:
                return 1
            else:
                return 2
        else:
            if stance1 != stance2: # Stance is valid when different, because this is point
                return 3
            else:
                return 4
    else:
        return 6


def run_eval(split,
             scorer: Callable[[Passage, List[Passage]], List[NamedNumber]],
             condition: EvalCondition):
    problems, candidate_d = prepare_eval_data(split)
    debug = False

    problems = problems[:100]
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    correctness = []
    fail_type_count = Counter()
    for query, problem in zip(payload, problems):
        p = problem
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate = list([candidate_d[x] for x in candidate_ids])
        scores: List[NamedNumber] = scorer(query, candidate)
        best_idx = get_max_idx(scores)
        pred_item: Passage = candidate[best_idx]
        gold_id = p.text2.id
        pred_id = pred_item.id
        correct = gold_id == pred_id
        content_equal = (p.text2.text == pred_item.text)
        correct = correct or content_equal
        gold_idx_l = list([idx for idx, c in enumerate(candidate) if c.id == gold_id])
        gold_idx = gold_idx_l[0] if gold_idx_l else None
        gold_score = scores[gold_idx] if gold_idx_l else None

        if not correct and debug:
            print("-------------------", correct, content_equal)
            print("QUERY:", p.text1.id)
            print(p.text1.text)
            print("GOLD:", p.text2.id)
            print(p.text2.text)
            print("PRED:", pred_item.id)
            # print(scores[best_idx].name)
            print(pred_item.text)
            print("RATIONALE: ", scores[best_idx].__float__(), scores[best_idx].name)
            if gold_score is not None:
                print("GOLD RATIONALE:", gold_score.__float__(), gold_score.name)

            t = failure_type(p.text2.id.id, pred_item.id.id)
            fail_type_count[t] += 1

        correctness.append(correct)
    avg_p_at_1 = average(correctness)
    #print(fail_type_count)
    return avg_p_at_1



def collect_failure(split,
            scorer: Callable[[Passage, List[Passage]], List[NamedNumber]],
             condition: EvalCondition):
    problems, candidate_pool_d = prepare_eval_data(split)

    problems = problems[:100]
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    for query, problem in zip(payload, problems):
        p = problem
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate: List[Passage] = list([candidate_pool_d[x] for x in candidate_ids])
        scores: List[NamedNumber] = scorer(query, candidate)
        best_idx = get_max_idx(scores)
        pred_item: Passage = candidate[best_idx]
        gold_id = p.text2.id
        pred_id = pred_item.id
        correct = gold_id == pred_id
        content_equal = (p.text2.text == pred_item.text)
        correct = correct or content_equal
        gold_idx_l = list([idx for idx, c in enumerate(candidate) if c.id == gold_id])
        gold_idx = gold_idx_l[0] if gold_idx_l else None
        gold_score = scores[gold_idx] if gold_idx_l else None
        if not correct :
            e = p.text1.text, p.text2.text, pred_item.text
            yield e


def is_correct(p: ArguDataPoint, c: Passage) -> bool:
    gold_id = p.text2.id
    correct: bool = gold_id == c.id
    content_equal: bool = (p.text2.text == c.text)
    correct = correct or content_equal
    return correct


def pairwise_candidate_gen(split,
                           condition: EvalCondition,
                           ) -> Iterable[Tuple[ArguDataPoint, Passage, bool]]:
    problems, candidate_pool_d = prepare_eval_data(split)

    debug = False
    if debug:
        problems = problems[:100]
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    for query, problem in zip(payload, problems):
        p = problem
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate: List[Passage] = list([candidate_pool_d[x] for x in candidate_ids])
        for c in candidate:
            correct = argu_is_correct(p, c)
            yield p, c, correct


def argu_is_correct(p: ArguDataPoint, c: Passage):
    gold_id = p.text2.id
    correct: bool = gold_id == c.id
    content_equal: bool = (p.text2.text == c.text)
    correct = correct or content_equal
    return correct


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


