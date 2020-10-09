from typing import List, Dict, Tuple

from arg.counter_arg.eval import EvalCondition
from arg.counter_arg.qck_datagen import load_base_resource
from arg.qck.decl import QCKCandidate


def main():
    split = NotImplemented


def get_pos_neg_docs(split) -> List[Tuple[str, List[str], List[str]]]:
    candidate_dict, correct_d = load_base_resource(EvalCondition.EntirePortalCounters, split)
    correct_d: Dict[Tuple[str, str], bool] = correct_d
    candidate_dict: Dict[str, List[QCKCandidate]] = candidate_dict

    output: List[Tuple[str, List[str], List[str]]] = []
    for q_id, candidate_id_list in candidate_dict.items():
        pos_docs = []
        neg_docs = []
        for c in candidate_id_list:
            cid = c.id
            if correct_d[q_id, cid]:
                pos_docs.append(c.text)
            else:
                neg_docs.append(c.text)

        if len(neg_docs) != len(candidate_id_list) - 1:
            print("neg/all = {}/{}".format(len(neg_docs), len(candidate_id_list)))

        e = q_id, pos_docs, neg_docs
        output.append(e)
    return output






if __name__ == "__main__":
    main()