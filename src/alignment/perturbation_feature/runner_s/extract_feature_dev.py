from typing import List

from bert_api.task_clients.nli_interface.nli_interface import NLIInput, get_nli_cache_client
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.extract_feature import perturbation_enum, RankList
from alignment.nli_align_path_helper import load_mnli_rei_problem
from misc_lib import tprint


def eval_all_perturbations(nli_client, problems):
    hash_key_set = set()
    for idx, p in enumerate(problems):
        tprint("problem {}".format(idx))
        ri_list: List[RankList] = perturbation_enum(p.seg_instance)

        for item in ri_list:
            for _, si_list in item.rank_item_list:
                todo = []
                for si in si_list:
                    nli_input = NLIInput(si.text2, si.text1)
                    hash_key = nli_input.str_hash()
                    if hash_key not in hash_key_set:
                        hash_key_set.add(hash_key)
                        todo.append(nli_input)

                nli_client.predict(todo)


def main():
    dataset = "train"
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)
    nli_client = get_nli_cache_client("localhost")
    eval_all_perturbations(nli_client, problems)


if __name__ == "__main__":
    main()


