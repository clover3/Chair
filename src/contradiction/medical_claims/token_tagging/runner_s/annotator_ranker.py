import os
from typing import List, Dict, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF2, \
    make_ranked_list_w_solver_if2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri1_problem_info_json, AlamriProblem
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from cpath import output_path
from list_lib import index_by_fn
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
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
            qrel_entries: Dict[DocID, int] = self.qrel[query_id]
            def get_score(doc_id)-> float:
                return 1 if doc_id in qrel_entries else 0

            scores: List[float] = [get_score(str(idx)) for idx, _ in enumerate(tokens)]
            return scores

        return get_answer("prem", text1_tokens), get_answer("hypo", text2_tokens)


def make_ranked_list(save_name, problems: List[AlamriProblem], source_qrel_path, tag_type):
    qrel: QRelsDict = load_qrels_structured(source_qrel_path)
    solver = AnnotatorSolver(qrel, problems, tag_type)
    run_name = "annotator"
    save_path = get_save_path(save_name)
    all_ranked_list = make_ranked_list_w_solver_if2(problems, run_name, save_path, tag_type, solver)
    write_trec_ranked_list_entry(all_ranked_list, save_path)


def make_ranked_list_old(problems, qrel, run_name, tag_type):
    all_ranked_list = []
    for p in problems:
        todo = [
            (p.text1, 'prem'),
            (p.text2, 'hypo'),
        ]

        for text, sent_name in todo:
            tokens = text.split()
            query_id = get_query_id(p.group_no, p.inner_idx, sent_name, tag_type)
            if query_id not in qrel:
                print("Query {} not found".format(query_id))
                continue
            qrel_entries: Dict[DocID, int] = qrel[query_id]

            def get_score(doc_id) -> float:
                return 1 if doc_id in qrel_entries else 0

            doc_id_score_list: List[Tuple[str, float]] \
                = [(str(idx), get_score(str(idx))) for idx, _ in enumerate(tokens)]

            ranked_list = scores_to_ranked_list_entries(doc_id_score_list, run_name, query_id)
            all_ranked_list.extend(ranked_list)


def main():
    info_d = load_alamri1_problem_info_json()
    qrel_path = os.path.join(output_path, "alamri_annotation1", "label", "worker_Q.qrel")
    tag_type = "conflict"
    save_name = "annotator_q_" + tag_type
    make_ranked_list(save_name, info_d, qrel_path, tag_type)


if __name__ == "__main__":
    main()