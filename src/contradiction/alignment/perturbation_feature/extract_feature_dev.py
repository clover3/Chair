from typing import List, Iterable, Callable, Dict, Tuple, Set

from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance
from contradiction.alignment.extract_feature import perturbation_enum, RankList
from contradiction.alignment.nli_align_path_helper import load_mnli_rei_problem


def main():
    dataset = "dev"
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)

    hash_key_set = set()
    for p in problems:
        ri_list: List[RankList] = perturbation_enum(p.seg_instance)

        for item in ri_list:
            for _, si_list in item.rank_item_list:
                for si in si_list:
                    hash_key = NLIInput(si.text2, si.text1).str_hash()
                    hash_key_set.add(hash_key)

    print("{} runs, {} per problem".format(
        len(hash_key_set),
        len(hash_key_set) / len(problems)
    ))

if __name__ == "__main__":
    main()


