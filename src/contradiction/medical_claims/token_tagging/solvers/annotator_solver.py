from typing import List, Dict, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from list_lib import index_by_fn
from trec.types import QRelsDict, DocID


class AnnotatorSolver(TokenScoringSolverIF2):
    def __init__(self, qrel: QRelsDict, problems: List[AlamriProblem], tag_type):
        self.tag_type = tag_type
        self.qrel = qrel
        self.problems_d: Dict[int, AlamriProblem] = index_by_fn(lambda p: p.data_id, problems)

    def solve(self, data_id: int, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        p = self.problems_d[data_id]

        def get_answer(sent_name, tokens):
            query_id = get_query_id(p.group_no, p.inner_idx, sent_name, self.tag_type)
            try:
                qrel_entries: Dict[DocID, int] = self.qrel[query_id]
            except KeyError:
                print("No judges for {}".format(query_id))
                qrel_entries = {}

            def get_score(doc_id) -> float:
                return 1 if doc_id in qrel_entries else 0

            scores: List[float] = [get_score(str(idx)) for idx, _ in enumerate(tokens)]
            return scores

        return get_answer("prem", text1_tokens), get_answer("hypo", text2_tokens)