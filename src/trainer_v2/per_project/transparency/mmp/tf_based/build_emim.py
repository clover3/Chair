import math
from cpath import output_path
from misc_lib import path_join
import pickle
from collections import Counter

from trainer_v2.per_project.transparency.misc_common import save_tsv


def log_odd(rel_tf, rel_sum, bg_tf, bg_sum):
    return math.log(rel_tf / rel_sum) - math.log(bg_tf / bg_sum)


def emim(rel_tf, rel_sum, bg_tf, bg_sum):
    # rel_tf (qt, dt)'s appearance in relevant documents
    # rel_sum (qt, dt')
    return rel_tf * math.log(bg_sum * rel_tf / (rel_sum * bg_tf))


def main():
    common_dir = path_join(output_path, "mmp", "lucene_krovetz")
    tf_path = path_join(common_dir, "tf.pkl")
    bg_tf = pickle.load(open(tf_path, "rb"))
    bg_sum = sum(bg_tf.values())

    rel_co_tf_path = path_join(common_dir, "rel_tf.pkl")
    tf_co_rel = pickle.load(open(rel_co_tf_path, "rb"))

    rel_dl_sum_per_q_term = Counter()
    for (qt, dt), cnt in tf_co_rel.items():
        rel_dl_sum_per_q_term[qt] += cnt
    cut = 10

    print("Emim cut={}".format(cut))
    emim_scores = []
    for (qt, dt), cnt in tf_co_rel.items():
        v = emim(cnt, rel_dl_sum_per_q_term[qt], bg_tf[dt], bg_sum)
        if v > cut:
            emim_scores.append((qt, dt, v))
    save_tsv(emim_scores, path_join(common_dir, f"emim_{cut}.tsv"))

    # emim_path = path_join(common_dir, "emim.pkl")
    # pickle.dump(emim_scores, open(emim_path, "wb"))


if __name__ == "__main__":
    main()
