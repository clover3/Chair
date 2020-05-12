from collections import Counter
from typing import List, Dict, Tuple

from arg.perspectives.basic_analysis import get_candidates
from arg.perspectives.declaration import PerspectiveCandidate
from arg.perspectives.load import get_claims_from_ids, load_test_claim_ids, load_dev_claim_ids, load_train_claim_ids, \
    claims_to_dict
from arg.perspectives.pc_tokenizer import PCTokenizer
from cache import save_to_pickle
from list_lib import dict_value_map, right, left
from misc_lib import group_by


def work():
    split = "train"
    assert split in ["train", "dev", "test"]

    tokenizer = PCTokenizer()
    d_ids = list({
        "train": load_train_claim_ids(),
        "dev": load_dev_claim_ids(),
        "test": load_test_claim_ids()
    }[split])
    claims = get_claims_from_ids(d_ids)
    claim_d = claims_to_dict(claims)

    print(len(claims), " claims")
    do_balance = False
    all_data_points: List[PerspectiveCandidate] = get_candidates(claims, do_balance)

    grouped: Dict[str, List] = group_by(all_data_points, lambda x: x.cid)

    def get_frequency_per_class(datapoints: List[PerspectiveCandidate]):
        pos_text = []
        neg_text = []
        for dp in datapoints:
            tokens = tokenizer.tokenize_stem(dp.p_text)
            tf = Counter(tokens)
            dl = sum(tf.values())
            tf_rel = {k: v/dl for k, v in tf.items()}

            if dp.label == "1":
                pos_text.append(tf_rel)
            elif dp.label == "0":
                neg_text.append(tf_rel)
            else:
                assert False

        def accumulate(tf_list: List[Dict]):
            out_c = Counter()
            n = len(tf_list)
            for tf in tf_list:
                for k, v in tf.items():
                    out_c[k] += v / n

            return out_c

        pos_avg_tf = accumulate(pos_text)
        neg_avg_tf = accumulate(neg_text)
        return pos_avg_tf, neg_avg_tf

    class_freq: Dict[str, Tuple[Counter, Counter]] = dict_value_map(get_frequency_per_class, grouped)

    save_to_pickle(class_freq, "per_claim_class_word_tf_{}".format(split))

    def normalize(s_list: List[float]) -> List[float]:
        m = sum(s_list)
        return list([s/m for s in s_list])


    pos_prob_dict = {}
    neg_prob_dict = {}

    for cid, info in class_freq.items():
        pos, neg = info
        all_words = set(pos.keys())
        all_words.update(neg.keys())

        info = []
        for word in all_words:
            score = pos[word] - neg[word]
            info.append((word, score))

        pos_scores = list([(w, s) for w, s in info if s > 0])
        neg_scores = list([(w, s) for w, s in info if s < 0])


        def normalize_right(pair_list):
            right_scores = normalize(right(pair_list))
            return list(zip(left(pair_list), right_scores))

        pos_prob_dict[cid] = normalize_right(pos_scores)
        neg_prob_dict[cid] = normalize_right(neg_scores)

    save_to_pickle(pos_prob_dict, "pc_pos_word_prob_{}".format(split))
    save_to_pickle(neg_prob_dict, "pc_neg_word_prob_{}".format(split))


if __name__ == "__main__":
    work()

