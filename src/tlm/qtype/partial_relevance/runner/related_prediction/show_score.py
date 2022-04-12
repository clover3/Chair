from typing import List, Dict

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from list_lib import index_by_fn
from alignment.data_structure.print_helper import rei_to_text
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer


#

def visualize_table(dataset, method):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    answers = load_related_eval_answer(dataset, method)
    print("{} problems {} answers".format(len(problems), len(answers)))
    tokenizer = get_tokenizer()
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problems)
    for a in answers:
        p: RelatedEvalInstance = pid_to_p[a.problem_id]
        if p.score < 0.5:
            continue
        print(rei_to_text(tokenizer, p))

        for seg_idx in [0, 1]:
            n_over_09 = len([s > 0.9 for s in a.contribution.table[seg_idx]])
            n_seg = len(a.contribution.table[seg_idx])
            scores = np.array(a.contribution.table[seg_idx])
            rank = np.argsort(scores)[::-1]
            seg_s = ids_to_text(tokenizer, p.seg_instance.text1.get_tokens_for_seg(seg_idx))
            print("For segment {} ({})".format(seg_idx, seg_s))
            print("{} of {} segments got over 0.9".format(n_over_09, n_seg))
            for i in rank[:10]:
                tokens_ids = p.seg_instance.text2.get_tokens_for_seg(i)
                word = ids_to_text(tokenizer, tokens_ids)
                score = scores[i]
                print("{0}: {1:.2f}".format(word, score))

        input("enter to continue")


def main():
    visualize_table("dev", "gradient")


if __name__ == "__main__":
    main()