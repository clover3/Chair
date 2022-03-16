from collections import Counter
from typing import List, Dict

from list_lib import index_by_fn
from tlm.qtype.partial_relevance.complement_path_data_helper import load_complements
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_dev_problems


def partial_relevant_rate(problem_list: List[RelatedEvalInstance],
                          complement_list: List[ComplementSearchOutput],
                         ):
    pid_to_c: Dict[str, ComplementSearchOutput] = index_by_fn(lambda e: e.problem_id, complement_list)

    counter = Counter()
    for p in problem_list:
        c: ComplementSearchOutput = pid_to_c[p.problem_id]
        f_rel = p.score >= 0.5
        f_part_rel = bool(c.complement_list)
        if f_rel and not f_part_rel:
            print(" ".join(p.query_info.out_s_list))
        sig = (f_rel, f_part_rel)
        counter[sig] += 1
    print(counter)


# Runs eval for Related against full query
def main():
    problems: List[RelatedEvalInstance] = load_dev_problems()
    complements = load_complements()
    partial_relevant_rate(problems, complements)


if __name__ == "__main__":
    main()
