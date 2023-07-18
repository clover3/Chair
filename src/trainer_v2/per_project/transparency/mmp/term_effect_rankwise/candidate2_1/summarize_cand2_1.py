from collections import Counter

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from iter_util import load_jsonl
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import term_align_candidate2_score_path


def load_term_pair_candidates(job_no) -> List[Dict]:
    save_path = path_join(output_path, "msmarco", "passage",
                          "candidate2_1_building", f"{job_no}.jsonl")
    items = load_jsonl(save_path)
    return items


def main():
    known_scores = {}
    save_path = term_align_candidate2_score_path()
    entries = list(tsv_iter(save_path))
    for q_term, d_term, score in entries:
        known_scores[q_term, d_term] = score

    items = []
    for job_no in range(100):
        items.extend(load_term_pair_candidates(job_no))

    tokenizer = get_tokenizer()

    def id_to_term(term_id):
        return tokenizer.convert_ids_to_tokens([term_id])[0]

    n_max = 0
    max_q = ""
    counter = Counter()
    for t in items:
        q_term = id_to_term(t['q_term'])
        term_score_pairs = t['matching']
        n_pos = 0
        for term, score in term_score_pairs:
            if score > 0:
                d_term = id_to_term(term)
                n_pos += 1
                if (q_term, d_term) in known_scores:
                    counter["pos seen"] += 1
                else:
                    counter["pos unseen"] += 1

        if n_pos:
            counter["pos_exists"] += 1
        counter["pos"] += n_pos
        counter["n query"] += 1

        if n_pos > n_max:
            n_max = n_pos
            max_q = q_term

    print(counter)
    print(f"{max_q} has maximum of {n_max} matches")


if __name__ == "__main__":
    main()
