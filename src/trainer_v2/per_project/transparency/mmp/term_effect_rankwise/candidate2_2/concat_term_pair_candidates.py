from typing import List, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from iter_util import load_jsonl
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.combine_inference import \
    filter_sharp_sharp_subword_and_drop_score, convert_term_pair_candidate_ids_to_terms
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_2.path_helper import \
    get_candidate2_2_term_pair_candidate_building_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper, \
    get_cand2_2_path_helper


def load_term_pair_candidate_over_100_jobs() -> List[Dict]:
    items = []
    for job_no in range(100):
        save_path = get_candidate2_2_term_pair_candidate_building_path(job_no)
        items_per_job = load_jsonl(save_path)
        items.extend(items_per_job)
    return items


def main():
    items = load_term_pair_candidate_over_100_jobs()
    tokenizer = get_tokenizer()
    all_candidates: List[Tuple[str, str, float]]\
        = convert_term_pair_candidate_ids_to_terms(tokenizer, items)
    config: MMPGAlignPathHelper = get_cand2_2_path_helper()
    selected = filter_sharp_sharp_subword_and_drop_score(all_candidates)
    save_tsv(selected, config.per_pair_candidates.candidate_pair_path)



if __name__ == "__main__":
    main()