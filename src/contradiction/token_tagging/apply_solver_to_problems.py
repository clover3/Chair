from typing import List, Callable, Tuple

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_scores_to_trec_entries
from alignment.base_ds import TextPairProblem
from misc_lib import TEL
from trec.trec_parse import write_trec_ranked_list_entry


def make_ranked_list_w_problem2score(problems: List[TextPairProblem], run_name, save_path, tag_type,
                                     problem_to_scores: Callable[[TextPairProblem], Tuple[List[float], List[float]]]):
    all_ranked_list = []
    for p in TEL(problems):
        scores1, scores2 = problem_to_scores(p)

        def get_query_id_inner(sent_no):
            return f"{tag_type}_{p.problem_id}_{sent_no}"

        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('1'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('2'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)

    write_trec_ranked_list_entry(all_ranked_list, save_path)


def make_ranked_list_w_solver(problems: List[TextPairProblem], run_name, save_path, tag_type,
                              solver: TokenScoringSolverIF):
    def solve(text_pair_problem: TextPairProblem):
        tokens1 = text_pair_problem.text1.split()
        tokens2 = text_pair_problem.text2.split()
        return solver.solve(tokens1, tokens2)

    return make_ranked_list_w_problem2score(problems, run_name, save_path, tag_type, solve)


def make_ranked_list_w_batch_solver(problems: List[TextPairProblem],
                                    run_name, save_path, tag_type, solver: BatchTokenScoringSolverIF):
    all_ranked_list = []
    payload = []
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)

    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    for p, output in zip(problems, batch_output):
        def get_query_id_inner(sent_no):
            return f"{tag_type}_{p.problem_id}_{sent_no}"

        scores1, scores2 = output
        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('1'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('2'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)
    write_trec_ranked_list_entry(all_ranked_list, save_path)

