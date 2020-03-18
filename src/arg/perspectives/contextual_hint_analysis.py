



# TODO foreach of the claim
# TODO   get_relevant_unigrams_for_claim
# TODO   get ranked_list for claim
# TODO      for doc in ranked_list
# TODO         for all term,
# TODO         update tf,df of (term, controversy), (term,no controversy)
# TODO         update tf,df of (term)
# TODO
# TODO
# TODO
# TODO
from collections import Counter

import math
import nltk
from scipy.stats import ttest_ind

from arg.perspectives import es_helper
from arg.perspectives.clueweb_helper import ClaimRankedList, load_doc, load_tf, save_tf, preload_tf, \
    preload_docs
from arg.perspectives.context_analysis_routine import count_term_stat, feature_extraction
from arg.perspectives.load import load_dev_claim_ids, get_claims_from_ids, get_claim_perspective_id_dict
from cache import load_from_pickle
from datastore.interface import flush
from list_lib import lmap, foreach
from misc_lib import tprint, average
from models.classic.lm_counter_io import LMClassifier


def get_perspective(claim, candidate_k):
    cid = claim["cId"]
    claim_text = claim["text"]
    lucene_results = es_helper.get_perspective_from_pool(claim_text, candidate_k)
    perspectives = []
    for _text, _pid, _score in lucene_results:
        perspectives.append((_text, _pid, _score))
    return claim_text, perspectives


def lower_all(token_list):
    return list([t.lower() for t in token_list])


class UnaryLM:
    def __init__(self, p_w_dict):
        self.p_w_dict = p_w_dict

    def per_token_odd(self, token):
        if token not in self.p_w_dict:
            return 0

        return math.log(self.p_w_dict[token] + 1e-4)


def load_and_format_doc(doc_id):
    try:
        tf = load_tf(doc_id)
    except KeyError:
        try:
            tokens = load_doc(doc_id)
            tf = Counter(tokens)
            save_tf(doc_id, tf)
        except KeyError:
            print("doc {} not found".format(doc_id))
            raise

    token_set = set(tf.keys())
    return {'doc_id': doc_id,
            'tf_d': tf,
            'dl': sum(tf.values()),
            'tokens_set': token_set,
            }


def get_relevant_unigrams(perspectives):
    unigrams = set()
    tokens_list = [lower_all(nltk.word_tokenize(_text)) for _text, _pid, _score in perspectives]
    foreach(unigrams.update, tokens_list)
    return unigrams


def claim_language_model_property():
    dev_claim_ids = load_dev_claim_ids()
    claims = get_claims_from_ids(dev_claim_ids)
    all_ranked_list = ClaimRankedList()
    all_voca = set()
    candidate_k = 50
    for claim in claims:
        claim_text, perspectives = get_perspective(claim, candidate_k)
        print(claim_text)
        unigrams = get_relevant_unigrams(perspectives)
        ranked_list = all_ranked_list.get(str(claim['cId']))
        doc_ids = [t[0] for t in ranked_list]
        print("Loading documents")
        preload_tf(doc_ids)
        docs = lmap(load_and_format_doc, doc_ids)

        foreach(lambda doc: all_voca.update(doc['tokens_set']), docs)

        # check hypothesis
        # check_hypothesis(all_voca, cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf, ctf_cont,
        #                  ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont, unigrams)

        print("counting terms stat")

        lm_classifier = build_lm(docs, unigrams)

        for p_entry in perspectives:
            _text, _pid, _score = p_entry
            tokens = nltk.word_tokenize(_text)
            score = sum(lmap(lm_classifier.per_token_odd, tokens))
            print(_text, score)


def build_lm(docs, unigrams):
    cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, \
    clueweb_df, clueweb_tf, ctf_cont, ctf_ncont, \
    df_cont, df_ncont, tf_cont, tf_ncont = count_term_stat(docs, unigrams)
    if cdf_cont == 0 or cdf_ncont == 0:
        term_features = {}
    else:
        term_features = feature_extraction(cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf,
                                           ctf_cont,
                                           ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont, unigrams)
    prob_c = Counter()
    prob_nc = Counter()
    for term, items in term_features.items():
        (p1, p2), (_, _) = items
        if p1 == 0 and p2 == 0:
            continue
        prob_c[term] = p1
        prob_nc[term] = p2
    lm_classifier = LMClassifier(prob_c, prob_nc)
    return lm_classifier

cached_all_claim_lm_pairs = None


def load_all_claim_lm():
    all_claim_lm_pairs = []
    for job_id in range(0, 14):
        save_name = "dev_claim_{}".format(job_id)
        data = load_from_pickle(save_name)
        all_claim_lm_pairs.extend(data)
    return all_claim_lm_pairs


def get_lm_for_claim_from_cache(dummy, cid, dummy2):
    global cached_all_claim_lm_pairs
    if cached_all_claim_lm_pairs is None:
        cached_all_claim_lm_pairs = dict(load_all_claim_lm())
    return cached_all_claim_lm_pairs[cid]


def predict_single_claim_with_lm(all_ranked_list, claim):
    candidate_k = 50
    claim_text, perspectives = get_perspective(claim, candidate_k)
    unigrams = get_relevant_unigrams(perspectives)
    cid = claim['cId']

    #lm_classifier = get_lm_for_claim(all_ranked_list, cid, unigrams)
    lm_classifier = get_lm_for_claim_from_cache(all_ranked_list, cid, unigrams)
    lm_classifier.smoothing = 0.9
    #lm_classifier = UnaryLM(lm_classifier.P_w_NC_dict)

    print(claim_text)
    prediction_list = []
    alpha = 0
    for idx, p_entry in enumerate(perspectives):
        _text, _pid, _score = p_entry
        tokens = nltk.word_tokenize(_text)
        lm_score = sum(lmap(lm_classifier.per_token_odd, tokens))
        lm_score_norm = -lm_score / 5
        _score = 40 / (idx+1)
        score = alpha * lm_score_norm + (1 - alpha) * _score
        rantionale_str = " rele={}({}) lm={} norm_lm={}".format(_score, idx, lm_score, lm_score_norm)
        pred = {
            'cid': cid,
            'claim_text': claim_text,
            'pid':_pid,
            'perspective_text': _text,
            'lm_score':lm_score,
            'score': score,
            'rationale': rantionale_str
        }
        prediction_list.append(pred)
    flush()
    prediction_list.sort(key=lambda x:x['score'], reverse=True)
    return prediction_list


def get_lm_for_claim(all_ranked_list, cid, unigrams):
    ranked_list = all_ranked_list.get(str(cid))
    doc_ids = [t[0] for t in ranked_list]
    tprint("Loading document")
    preload_docs(doc_ids)
    preload_tf(doc_ids)
    docs = lmap(load_and_format_doc, doc_ids)
    tprint("building clm document")
    lm_classifier = build_lm(docs, unigrams)
    return lm_classifier


# input : claims, top_k
# output : List(cid, List[dict])
def predict_with_lm(claims, top_k):
    all_ranked_list = ClaimRankedList()

    def predict_each(claim):
        prediction_list = predict_single_claim_with_lm(all_ranked_list, claim)
        return claim['cId'], prediction_list[:top_k]

    return lmap(predict_each, claims)


def perspective_lm_correlation():
    d_ids = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    top_k = 20
    gold = get_claim_perspective_id_dict()
    predictions = predict_with_lm(claims, top_k)

    avg_pos_list = []
    avg_neg_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        claim_text = prediction_list[0]['claim_text']

        pos_list = []
        neg_list = []
        print("Claim: ", claim_text)
        for prediction in prediction_list:
            pid = prediction['pid']
            valid = False
            for cluster in gold_pids:
                if pid in cluster:
                    valid = True
                    break
            print("{0} {1:.2f} {2}".format(valid, prediction['lm_score'], prediction['perspective_text']))
            if not valid:
                neg_list.append(prediction['lm_score'])
            else:
                pos_list.append(prediction['lm_score'])

        if pos_list and neg_list:
            pos_score = average(pos_list)
            neg_score = average(neg_list)
            avg_pos_list.append(pos_score)
            avg_neg_list.append(neg_score)

    diff, p = ttest_ind(avg_pos_list, avg_neg_list)
    print("pos", average(avg_pos_list), "neg", average(avg_neg_list))
    print("pos", avg_pos_list)
    print("neg", avg_neg_list)
    print(diff, p)


if __name__ == "__main__":
    perspective_lm_correlation()
