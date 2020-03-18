import os
import pickle
from collections import Counter

from scipy.special import softmax

from arg.claim_building.count_ngram import build_ngram_lm_from_tokens_list, merge_subword
from arg.stance_build import build_uni_lm_from_tokens_list, display
from cache import save_to_pickle, load_from_pickle
from list_lib import lmap
from misc_lib import get_dir_files
from tlm.ukp.data_gen.ukp_gen_selective import load_ranked_list
from tlm.ukp.sydney_data import ukp_load_tokens_for_topic


def enum_docs_and_stance():
    topic = "abortion"
    summary_path = "/mnt/nfs/work3/youngwookim/data/stance/clueweb12_10000_pred_ex_summary_w_logit"
    relevance_list_path = "/home/youngwookim/work/ukp/relevant_docs/clueweb12"
    all_tokens = ukp_load_tokens_for_topic(topic)
    all_ranked_list = load_ranked_list(relevance_list_path)
    for file_path in get_dir_files(summary_path):
        if topic not in file_path:
            continue
        file_name = os.path.basename(file_path)
        predictions = pickle.load(open(file_path, "rb"))
        for doc_idx, preds in predictions:
            doc_id, rank, score = all_ranked_list[file_name][doc_idx]
            doc = all_tokens[doc_id]
            yield doc, preds


def majority(build_lm_from_tokens_list, save_name):
    tf0 = Counter()
    tf1 = Counter()
    tf2 = Counter()
    for doc, preds in enum_docs_and_stance():
        assert len(preds) == len(doc)
        cnt_stance1 = 0
        cnt_stance2 = 0
        for sent, pred in zip(doc, preds):
            probs = softmax(pred)
            if probs[1] > 0.5:
                cnt_stance1 += 1
            elif probs[2] > 0.5:
                cnt_stance2 += 1

        if cnt_stance1 > cnt_stance2:
            stance = 1
        elif cnt_stance2 > cnt_stance1:
            stance = 2
        else:
            stance = 0

        if stance > 0:
            tf = build_lm_from_tokens_list(doc)
            [tf0, tf1, tf2][stance].update(tf)

    result = tf0, tf1, tf2
    save_to_pickle(result, save_name)
    display(tf1, tf2, "favor", "against")



def only_sent():
    tf0 = Counter()
    tf1 = Counter()
    tf2 = Counter()
    for doc, preds in enum_docs_and_stance():
        assert len(preds) == len(doc)
        for sent, pred in zip(doc, preds):
            probs = softmax(pred)
            stance = 0
            if probs[1] > 0.5:
                stance = 1
            elif probs[2] > 0.5:
                stance = 2

            if stance > 0:
                tf = build_uni_lm_from_tokens_list([sent])
                [tf0, tf1, tf2][stance].update(tf)

    result = tf0, tf1, tf2
    save_to_pickle(result, "only_sent")

    display(tf1, tf2, "favor", "against")


def predict(pred):
    probs = softmax(pred)
    stance = 0
    if probs[1] > 0.5:
        stance = 1
    elif probs[2] > 0.5:
        stance = 2
    return stance


def near_by():
    tf0 = Counter()
    tf1 = Counter()
    tf2 = Counter()
    window = 3
    for doc, preds in enum_docs_and_stance():
        assert len(preds) == len(doc)
        predictions = lmap(predict, preds)
        for idx, sent in enumerate(doc):
            tf = build_uni_lm_from_tokens_list([sent])
            for j in range(idx-window, idx+window+1):
                if j < 0 or len(doc) <= j:
                    continue
                stance = predictions[j]
                [tf0, tf1, tf2][stance].update(tf)

    result = tf0, tf1, tf2
    save_to_pickle(result, "near_by")

    display(tf1, tf2, "favor", "against")


def build_ngram_lm_in_word_level(doc, n):
    doc = [merge_subword(s) for s in doc]
    return build_ngram_lm_from_tokens_list(doc, n)


def show_from_pickle():
    result = load_from_pickle("majority")
    tf0, tf1, tf2 = result
    display(tf1, tf2, "favor", "against")


if __name__ == "__main__":
    majority(lambda x: build_ngram_lm_in_word_level(x, 3), "majority_3gram")
