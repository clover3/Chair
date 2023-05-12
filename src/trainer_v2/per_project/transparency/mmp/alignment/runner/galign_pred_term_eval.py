from transformers import AutoTokenizer

from cache import load_from_pickle
from misc_lib import get_second
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.galign_label import compute_gain_10K_when
import numpy as np

def get_gold():
    term_gain = compute_gain_10K_when()
    term_gain.sort(key=get_second)
    # This term gain is rank change. Lower (negative) the better
    pos_terms = []
    for term, score in term_gain[:1000]:
        if score < 0:
            pos_terms.append(term)

    neg_terms = []
    for term, score in term_gain[-1000:]:
        if score > 0:
            neg_terms.append(term)

    middle_terms = {}
    for term, score in term_gain:
        if term not in pos_terms and term not in neg_terms:
            middle_terms[term] = score

    return pos_terms, neg_terms, middle_terms


def re_order(doubled_pred: np.array, batch_size):
    seen_batch_size = batch_size * 2
    cursor = 0
    valid_output = []
    while cursor < len(doubled_pred):
        num_remain = len(doubled_pred) - cursor
        if num_remain >= seen_batch_size:
            cur_seen_batch_size = seen_batch_size
        else:
            cur_seen_batch_size = num_remain

        cur_batch_size = int(cur_seen_batch_size / 2)
        batch = doubled_pred[cursor: cursor + cur_seen_batch_size]
        a = batch[:cur_batch_size]
        b = batch[cur_batch_size:]
        error = np.abs(a-b)
        if not np.all(np.less(error, 1e-2)):
            print("Error is large", error)
            raise Exception()
        valid_output.append(a)

        cursor += cur_seen_batch_size

    return np.concatenate(valid_output, axis=0)


def main():
    c_log.info("Loading pickle")
    output = load_from_pickle("galign_pred")
    align_pred_raw = output['align_probe']['all_concat']
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    c_log.info("Re ordering")
    align_pred = re_order(align_pred_raw, 16)
    entries = []
    for term_id in range(100, 10000):
        idx = (term_id - 100)
        score = align_pred[idx]
        term = rev_vocab[term_id]
        entries.append((term, score))

    entries.sort(key=get_second, reverse=True)
    c_log.info("Get gold")
    pos_terms, neg_terms, middle_terms = get_gold()

    for term, score in entries:
        if term in pos_terms:
            label = "Pos"
        elif term in neg_terms:
            label = "Neg"
        else:
            try:
                label = "Middle: {}".format(middle_terms[term])
            except KeyError:
                label = "not in label"

        print(term, score, label)


if __name__ == "__main__":
    main()