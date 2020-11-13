from typing import Tuple, Dict, List

from arg.counter_arg.eval import prepare_eval_data, get_eval_payload_from_dp, retrieve_candidate, argu_is_correct
from arg.counter_arg.header import Passage, ArguDataID
from arg.counter_arg.tf_datagen.qck_common import problem_to_qck_query, passage_to_candidate
from arg.qck.decl import QCKCandidate
from list_lib import lmap


def load_base_resource(condition, split) -> Tuple[Dict[str, List[QCKCandidate]], Dict[Tuple[str, str], bool]]:
    problems, candidate_pool_d = prepare_eval_data(split)
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    correct_d = {}
    candidate_dict: Dict[str, List[QCKCandidate]] = dict()
    for query, problem in zip(payload, problems):
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate: List[Passage] = list([candidate_pool_d[x] for x in candidate_ids])
        qck_query = problem_to_qck_query(problem)
        qck_candidate_list: List[QCKCandidate] = lmap(passage_to_candidate, candidate)
        candidate_dict[qck_query.query_id] = qck_candidate_list

        correct_list = list([argu_is_correct(problem, c) for c in candidate])
        for c, correct in zip(qck_candidate_list, correct_list):
            pair_id = qck_query.query_id, c.id
            correct_d[pair_id] = correct
    return candidate_dict, correct_d