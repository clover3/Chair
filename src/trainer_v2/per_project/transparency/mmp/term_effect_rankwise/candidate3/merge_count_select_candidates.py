import csv
import logging
import math
import os.path
from collections import Counter
from cache import load_pickle_from, save_to_pickle
from misc_lib import group_by, get_first, get_second
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignConfig, \
    MMPGAlignPathHelper, get_mmp_galign_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train


def main():
    def iter_pickle_path():
        for i in get_valid_mmp_partition_for_train():
            save_path = config.get_sub_dir_partition_path("tf_rel_count", str(i))
            yield save_path

    config: MMPGAlignPathHelper = get_mmp_galign_path_helper()
    freq_q_terms = set(config.load_freq_q_terms())
    rel_count_acc = Counter()
    c_log.setLevel(logging.DEBUG)
    for save_path in iter_pickle_path():
        if not os.path.exists(save_path):
            print("Missing ", save_path)

    tf_acc = Counter()
    for save_path in iter_pickle_path():
        try:
            tf, rel_count = load_pickle_from(save_path)
            for d_term, rel_cnt in tf.items():
                tf_acc[d_term] += rel_cnt

            for k, v in rel_count.items():
                q_term, d_term = k
                if q_term in freq_q_terms:
                    rel_count_acc[k] += v
        except FileNotFoundError as e:
            print(e)

    print("Done")
    collection_size = sum(tf_acc.values())
    entries = []
    for (q_term, d_term), rel_cnt in rel_count_acc.items():
        entries.append((q_term, d_term, rel_cnt))

    output = []
    top_k = 40
    for q_term, sub_entries in group_by(entries, get_first).items():
        n_term = sum([cnt for _, _, cnt in sub_entries])
        candidates = []
        for _, d_term, rel_cnt in sub_entries:
            p_w_rel = rel_cnt / n_term
            p_w_bg = tf_acc[d_term] / collection_size
            log_odd = math.log(p_w_rel) - math.log(p_w_bg)
            if rel_cnt > 2 and tf_acc[d_term] > 10:
                candidates.append({
                    'term': d_term,
                    'log_odd': log_odd,
                    'rel_cnt': rel_cnt,
                    'bg_cnt': tf_acc[d_term],
                })
        candidates.sort(key=lambda x: x['log_odd'], reverse=True)
        take_size = min(int(len(candidates) * 0.2), top_k)
        selected = candidates[:take_size] + candidates[-take_size:]
        for e in selected:
            d_term = e['term']
            log_odd = e['log_odd']
            rel_cnt = e['rel_cnt']
            bg_cnt = e['bg_cnt']
            output.append((q_term, d_term, log_odd, rel_cnt, bg_cnt))

    save_path = config.path_config.candidate_pair
    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    tsv_writer.writerows(output)


if __name__ == "__main__":
    main()
