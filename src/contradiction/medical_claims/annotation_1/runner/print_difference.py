import os
from typing import Dict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from cpath import output_path
from list_lib import index_by_fn
from misc_lib import path_join
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict, DocID


def main():
    # measure if the precision is better when the NLI prediction is correct.
    # I expect
    # If NLI prediction=contradiction, it will do good at 'conflict' prediction
    # If NLI prediction=neutral, it will do good at 'mismatch' prediction
    qrel_name = "worker_J.qrel"
    def load_qrel(worker):
        qrel_path = os.path.join(output_path, "alamri_annotation1", "label",  "worker_{}.qrel".format(worker))
        qrel: QRelsDict = load_qrels_structured(qrel_path)
        return qrel

    problems: List[AlamriProblem] = load_alamri_problem()
    problems_d = index_by_fn(lambda p: (p.group_no, p.inner_idx), problems)

    qrel1 = load_qrel("J")
    qrel2 = load_qrel("Q")

    for qid in qrel1:
        if qid in qrel2:
            group_no, problem_no, sent_type, tag = qid.split("_")
            problem = problems_d[int(group_no), int(problem_no)]
            entries1: Dict[DocID, int] = qrel1[qid]
            entries2: Dict[DocID, int] = qrel2[qid]
            text = {
                'prem': problem.text1,
                'hypo': problem.text2,
            }[sent_type]
            tokens = text.split()

            def get_printable(entries: Dict[DocID, int]):
                out_tokens = []
                for i, t in enumerate(tokens):
                    doc_id = str(i)
                    if doc_id in entries and entries[doc_id]:
                        out_tokens.append("[{}]".format(t))
                    else:
                        out_tokens.append(t)
                return " ".join(out_tokens)

            print(get_printable(entries1))
            print(get_printable(entries2))




if __name__ == "__main__":
    main()