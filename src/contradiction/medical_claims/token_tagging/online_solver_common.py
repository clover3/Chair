import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_scores_to_trec_entries
from misc_lib import TEL
from trec.trec_parse import write_trec_ranked_list_entry


class TokenScoringSolverIF(ABC):
    @abstractmethod
    def solve(self, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        pass

    def solve_from_text(self, text1: str, text2: str) -> Tuple[List[float], List[float]]:
        tokens1 = text1.split()
        tokens2 = text2.split()
        scores1, scores2 = self.solve(tokens1, tokens2)

        def check_length(scores, tokens):
            if len(scores) != len(tokens):
                print("WARNING number of scores ({}) doesn't match number of tokens ({})".format(len(scores), len(tokens)))

        check_length(scores1, tokens1)
        check_length(scores2, tokens2)
        return scores1, scores2


class TokenScoringSolverIF2(ABC):
    @abstractmethod
    def solve(self, data_id: int, text1_tokens: List[str], text2_tokens: List[str]) -> Tuple[List[float], List[float]]:
        pass

    def solve_from_text(self, data_id: int, text1: str, text2: str) -> Tuple[List[float], List[float]]:
        return self.solve(data_id, text1.split(), text2.split())


def make_ranked_list_w_solver(info_path, run_name, save_path, tag_type, solver: TokenScoringSolverIF):
    info_d = json.load(open(info_path, "r", encoding="utf-8"))

    all_ranked_list = []
    for data_id, info in info_d.items():
        text1 = info['text1']
        text2 = info['text2']
        scores1, scores2 = solver.solve_from_text(text1, text2)

        def get_query_id_inner(sent_name):
            return get_query_id(info['group_no'], info['inner_idx'], sent_name, tag_type)

        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)
    write_trec_ranked_list_entry(all_ranked_list, save_path)


def make_ranked_list_w_solver2(problems: List[AlamriProblem], run_name, save_path, tag_type,
                               solver: TokenScoringSolverIF):
    def problem_to_score_fn(p: AlamriProblem):
        return solver.solve_from_text(p.text1, p.text2)

    return make_ranked_list_w_solver3(problems, run_name, save_path, tag_type, problem_to_score_fn)


def make_ranked_list_w_solver_if2(problems: List[AlamriProblem], run_name, save_path, tag_type,
                                  solver: TokenScoringSolverIF2):
    def problem_to_score_fn(p: AlamriProblem):
        return solver.solve_from_text(p.data_id, p.text1, p.text2)

    return make_ranked_list_w_solver3(problems, run_name, save_path, tag_type, problem_to_score_fn)


def make_ranked_list_w_solver3(problems: List[AlamriProblem], run_name, save_path, tag_type,
                               problem_to_scores: Callable[[AlamriProblem], Tuple[List[float], List[float]]]):
    all_ranked_list = []
    for p in TEL(problems):
        scores1, scores2 = problem_to_scores(p)

        def get_query_id_inner(sent_name):
            return get_query_id(p.group_no, p.inner_idx, sent_name, tag_type)

        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)

    write_trec_ranked_list_entry(all_ranked_list, save_path)
