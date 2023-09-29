import json
from typing import List, Iterable, Callable, Dict, Tuple, Set

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.porting.save_eval_data import save_to_tsv
from iter_util import load_jsonl
from trainer_v2.per_project.transparency.misc_common import save_tsv
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


def convert_term_pair_candidate_ids_to_terms(tokenizer, items) -> List[Tuple[str, str, float]]:
    def id_to_term(term_id):
        return tokenizer.convert_ids_to_tokens([term_id])[0]

    all_candidates = []
    for t in items:
        q_term = id_to_term(t['q_term'])
        term_score_pairs = t['matching']
        for term_id, score in term_score_pairs:
            d_term = id_to_term(term_id)
            entry = q_term, d_term, score
            all_candidates.append(entry)
    return all_candidates


def main():
    items = load_term_pair_candidate_over_100_jobs()
    tokenizer = get_tokenizer()
    all_candidates: List[Tuple[str, str, float]]\
        = convert_term_pair_candidate_ids_to_terms(tokenizer, items)
    config: MMPGAlignPathHelper = get_cand2_2_path_helper()
    selected = []
    for q, d, s in all_candidates:
        if d[:2] == "##":
            pass
        else:
            selected.append((q, d))
    save_tsv(selected, config.per_pair_candidates.candidate_pair_path)


if __name__ == "__main__":
    main()