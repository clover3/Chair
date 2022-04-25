from typing import List

from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from alignment.extract_feature import eval_all_perturbations
from alignment.nli_align_path_helper import load_mnli_rei_problem


def main():
    dataset = "train"
    problems: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset)
    nli_client = get_nli_cache_client("localhost")
    eval_all_perturbations(nli_client, problems)


if __name__ == "__main__":
    main()


