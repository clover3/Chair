import os
from typing import List

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_scores_to_trec_entries
from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex, MNLIExEntry
from cpath import output_path
from misc_lib import TEL
from trec.trec_parse import write_trec_ranked_list_entry


def make_ranked_list_w_solver_mnli(problems: List[MNLIExEntry],
                                   solver: TokenScoringSolverIF,
                                   run_name, save_path):
    all_ranked_list = []
    for p in TEL(problems):
        scores1, scores2 = solver.solve_from_text(p.premise, p.hypothesis)

        def get_query_id_inner(sent_name):
            query_id = "{}_{}".format(p.data_id, sent_name)
            return query_id

        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)

    write_trec_ranked_list_entry(all_ranked_list, save_path)


def make_ranked_list_w_batch_solver(problems: List[MNLIExEntry],
                                    run_name, save_path, solver: BatchTokenScoringSolverIF):
    all_ranked_list = []
    payload = []
    for p in problems:
        input_per_problem = p.premise.split(), p.hypothesis.split()
        payload.append(input_per_problem)

    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    for p, output in zip(problems, batch_output):
        def get_query_id_inner(sent_name):
            query_id = "{}_{}".format(p.data_id, sent_name)
            return query_id

        scores1, scores2 = output
        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)
    write_trec_ranked_list_entry(all_ranked_list, save_path)


def get_save_path(save_name):
    save_path = os.path.join(output_path, "mnli_ex", "ranked_list", save_name + ".txt")
    return save_path


def solve_mnli_tag(split, run_name, solver: TokenScoringSolverIF, tag_type):
    problems: List[MNLIExEntry] = load_mnli_ex(split, tag_type)
    save_name = "{}_{}_{}".format(split, run_name, tag_type)
    save_path = get_save_path(save_name)
    make_ranked_list_w_solver_mnli(problems, solver, run_name, save_path)
