from abc import ABC, abstractmethod
from typing import List, Tuple

from attribution.attrib_types import TokenScores
from contradiction.mnli_ex.load_mnli_ex_data import MNLIExEntry


#
#
# def make_ranked_list_w_solver3(problems: List[MNLIExEntry], run_name, save_path, tag_type,
#                                                             problem_to_scores: Callable[[MNLIExEntry], Tuple[List[float], List[float]]]):
#     all_ranked_list = []
#     for p in TEL(problems):
#         scores1, scores2 = problem_to_scores(p)
#
#         def get_query_id_inner(sent_name):
#             return get_query_id(p.group_no, p.inner_idx, sent_name, tag_type)
#
#         rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
#         rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)
#
#         all_ranked_list.extend(rl1)
#         all_ranked_list.extend(rl2)
#
#     write_trec_ranked_list_entry(all_ranked_list, save_path)
#
#
# def make_ranked_list_w_solver2(problems: List[MNLIExEntry], run_name, save_path, tag_type,
#                                solver: TokenScoringSolverIF):
#     def problem_to_score_fn(p: MNLIExEntry):
#         return solver.solve_from_text(p.text1, p.text2)
#
#     return make_ranked_list_w_solver3(problems, run_name, save_path, tag_type, problem_to_score_fn)


class MNLIExSolver(ABC):
    @abstractmethod
    def explain(self, data: List[MNLIExEntry], target_label) -> List[Tuple[TokenScores, TokenScores]]:
        pass