from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set

from iter_util import load_jsonl


def filter_sharp_sharp_subword(all_candidates):
    selected = []
    for q, d, s in all_candidates:
        if d[:2] == "##":
            pass
        else:
            selected.append((q, d))
    return selected


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


def load_term_pair_candidate_over_100_jobs(dir_path) -> List[Dict]:
    items = []
    for job_no in range(100):
        save_path = path_join(dir_path, f"{job_no}.jsonl")
        items_per_job = load_jsonl(save_path)
        items.extend(items_per_job)
    return items
