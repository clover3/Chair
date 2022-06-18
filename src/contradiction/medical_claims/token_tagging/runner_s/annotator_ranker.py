import os
from typing import List, Dict, Tuple

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver_if2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri1_problem_info_json, AlamriProblem, \
    load_alamri_split
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.solvers.annotator_solver import AnnotatorSolver
from cpath import output_path
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import scores_to_ranked_list_entries
from trec.types import QRelsDict, DocID


def make_ranked_list(save_name, problems: List[AlamriProblem], source_qrel_path, tag_type):

    qrel: QRelsDict = load_qrels_structured(source_qrel_path)
    known_data_ids = set()
    for qid in qrel.keys():
        group_no, inner_idx, _, _ = qid.split("_")
        known_data_ids.add("{}_{}".format(group_no, inner_idx))

    # def known(p: AlamriProblem):
    #     return p.get_problem_id() in known_data_ids
    #
    # print("{} problems".format(len(problems)))
    # problems = list(filter(known, problems))
    # print("{} problems".format(len(problems)))

    solver = AnnotatorSolver(qrel, problems, tag_type)
    run_name = "annotator"
    save_path = get_save_path(save_name)
    make_ranked_list_w_solver_if2(problems, run_name, save_path, tag_type, solver)


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


def main2():
    info_d = load_alamri1_problem_info_json()
    problems = load_alamri_split('dev')
    qrel_path = os.path.join(output_path, "alamri_annotation1", "label", "worker_Q.qrel")
    tag_type = "conflict"
    save_name = "annotator_q_" + tag_type
    make_ranked_list(save_name, problems, qrel_path, tag_type)



def main():
    info_d = load_alamri1_problem_info_json()
    problems = load_alamri_split('dev')
    qrel_path = os.path.join(output_path, "alamri_annotation1", "label", "sbl.qrel.val")
    tag_type = "mismatch"
    save_name = "sbl_" + tag_type
    make_ranked_list(save_name, problems, qrel_path, tag_type)


if __name__ == "__main__":
    main()