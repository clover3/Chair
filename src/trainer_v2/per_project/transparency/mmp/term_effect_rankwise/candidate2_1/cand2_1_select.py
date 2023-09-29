import random

from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import get_second
from trainer_v2.per_project.transparency.misc_common import save_tsv
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_1.summarize_cand2_1 import \
    load_term_pair_candidates


def select_candidate_term_pars(candidate_set_name, size_config):
    n_max_pos = size_config['max_pos']
    n_hard_neg_sample = size_config['hard_neg_sample']
    n_random_neg_sample = size_config['random_neg_sample']
    items = load_term_pair_candidate_over_100_jobs()
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

        pos_to_use = pos_items[:n_max_pos]
        hard_neg_items = neg_items[:n_hard_neg_sample]
        random_neg_items = neg_items[n_random_neg_sample:]
        random.shuffle(random_neg_items)
        neg_to_use2 = random_neg_items[:n_random_neg_sample]
        selected_items = pos_to_use + hard_neg_items + neg_to_use2

        for d_term in selected_items:
            all_candidates.append((q_term, d_term))
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates",
        f"{candidate_set_name}.tsv")
    save_tsv(all_candidates, save_path)


def load_term_pair_candidate_over_100_jobs():
    items = []
    for job_no in range(100):
        items.extend(load_term_pair_candidates(job_no))
    return items


def main():
    # if the query has pos terms,
    #   cap the maximum pos terms to 20.
    #   Select the same number of neg terms from negative scored ones
    #   if the pairs are less than 10, select neg terms to match 20
    #       half as top-k, half as random from remain
    # if query has no pos d_term,
    #   select 20 top scores items.
    size_config = {
        'max_pos': 20,
        'hard_neg_sample': 10,
        'random_neg_sample': 10
    }
    candidate_set_name = 'candidate2_1'

    select_candidate_term_pars(candidate_set_name, size_config)


if __name__ == "__main__":
    main()
