from typing import List, Dict

from arg.counter_arg.eval import load_problems
from arg.counter_arg.header import ArguDataPoint
from arg.qck.decl import QKUnit
from cache import load_from_pickle


def main():
    split = "training"
    qk_list: List[QKUnit] = load_from_pickle("ca_qk_candidate_{}".format(split))
    problems: List[ArguDataPoint] = load_problems(split)

    q_id_to_problem: Dict[str, ArguDataPoint] = {p.text1.id.id: p for p in problems}

    for q, k_list in qk_list[:50]:
        print("------")
        problem = q_id_to_problem[q.query_id]
        print("<Query>")
        print(q.text)
        print("< Counter-argument >")
        print(problem.text2.text)
        doc_ids = list([k.doc_id for k in k_list])
        doc_ids = list(dict.fromkeys(doc_ids))
        for doc_id in doc_ids:
            print(doc_id)


if __name__ == "__main__":
    main()