import random

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import get_second
from trainer_v2.per_project.transparency.misc_common import save_tsv
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_1.summarize_cand2_1 import \
    load_term_pair_candidates


def select_candidate_term_pars(save_name):
    items = []
    for job_no in range(100):
        items.extend(load_term_pair_candidates(job_no))
    tokenizer = get_tokenizer()

    def id_to_term(term_id):
        return tokenizer.convert_ids_to_tokens([term_id])[0]

    all_candidates = []
    for t in items:
        q_term = id_to_term(t['q_term'])
        term_score_pairs = t['matching']
        term_score_pairs.sort(key=get_second, reverse=True)
        pos_items = []
        neg_items = []
        for term_id, score in term_score_pairs:
            term = id_to_term(term_id)
            if score > 0:
                pos_items.append(term)
            else:
                neg_items.append(term)

        for d_term in pos_items:
            all_candidates.append((q_term, d_term))
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates",
        f"{save_name}.tsv")
    save_tsv(all_candidates, save_path)


def main():
    candidate_set_name = 'candidate2_1_pos'
    select_candidate_term_pars(candidate_set_name)


if __name__ == "__main__":
    main()
