from typing import NamedTuple, List, Iterator

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_scores_to_trec_entries

from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from data_generator.NLI.enlidef import is_mnli_ex_target
from misc_lib import TEL
from trec.trec_parse import write_trec_ranked_list_entry


def parse_comma_sep_indices(s):
    tokens = s.split(",")
    tokens = [t for t in tokens if t]
    return list(map(int, tokens))


class NLIExEntry(NamedTuple):
    data_id: str
    premise: str
    hypothesis: str
    p_indices: List[int]
    h_indices: List[int]

    @classmethod
    def from_dict(cls, d):
        return NLIExEntry(
            d['data_id'],
            d['premise'],
            d['hypothesis'],
            parse_comma_sep_indices(d['p_indices']),
            parse_comma_sep_indices(d['h_indices']),
        )


def nli_ex_entry_to_sent_token_label(e: NLIExEntry, tag_type) -> Iterator[SentTokenLabel]:
    todo = [
        ("prem", e.p_indices, e.premise),
        ("hypo", e.h_indices, e.hypothesis)
    ]
    for sent_type, indices, text in todo:
        if is_mnli_ex_target(tag_type, sent_type):
            n_tokens = len(text.split())
            binary = [1 if i in indices else 0 for i in range(n_tokens)]
            yield SentTokenLabel(
                get_nli_ex_entry_qid(e, sent_type),
                binary
            )


def get_nli_ex_entry_qid(e, sent_type):
    query_id = "{}_{}".format(e.data_id, sent_type)
    return query_id


def make_ranked_list_w_solver_nli(problems: List[NLIExEntry],
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


def make_ranked_list_w_batch_solver(problems: List[NLIExEntry],
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