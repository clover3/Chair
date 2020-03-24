import numpy as np
from numpy import std, median

from arg.perspectives import es_helper
from arg.perspectives.build_feature import average_tf_over_docs
from arg.perspectives.collection_based.datapoint_strct import split_pos_neg, get_num_mention, get_tf_from_datapoint, \
    get_label
from cache import load_from_pickle
from galagos.interface import send_doc_queries, format_query_bm25
from list_lib import lmap
from misc_lib import average, split_7_3
from models.classic.lm_counter_io import LMClassifier
from sydney_clueweb.clue_path import get_first_disk


def learn_lm(feature_label_list):
    neg_insts, pos_insts = split_pos_neg(feature_label_list)

    def do_average(insts):
        return average_tf_over_docs(lmap(get_tf_from_datapoint, insts), len(insts))

    pos_lm = do_average(pos_insts)
    neg_lm = do_average(neg_insts)

    classifier = LMClassifier(pos_lm, neg_lm)

    xy = lmap(lambda x: (get_tf_from_datapoint(x), int(x[1])), feature_label_list)
    classifier.tune_alpha(xy)
    return classifier


def mention_num_classifier(train, val):
    neg_insts, pos_insts = split_pos_neg(train)
    print(len(train), len(pos_insts), len(neg_insts))

    n_mention_pos = lmap(get_num_mention, pos_insts)
    n_mention_neg = lmap(get_num_mention, neg_insts)

    all_x = lmap(get_num_mention, train)
    all_y = lmap(get_label, train)
    alpha = median(all_x)
    print(alpha)
    print("pos avg:", average(n_mention_pos), std(n_mention_pos))
    print("neg avg:", average(n_mention_neg), std(n_mention_neg))

    def eval(all_x, all_y):
        pred_y = lmap(lambda x: x > alpha, all_x)
        is_correct = np.equal(all_y, pred_y)
        return average(is_correct)

    print('train accuracy : ', eval(all_x, all_y))
    all_x = lmap(get_num_mention, val)
    all_y = lmap(get_label, val)
    print('val accuracy : ', eval(all_x, all_y))


def get_adjusted_score(data_point):
    n = get_num_mention(data_point)

    alpha = 14
    diff = n-alpha
    diff = min(10, diff)
    diff = max(-10, diff)
    print(n, diff, diff/5)
    return diff / 5


def predict_by_mention_num(claims, top_k):
    data_points = load_from_pickle("pc_train_features_binary")

    def get_key_from_feature(feature):
        return "{}_{}".format(feature['cid'], feature['pid'])

    datapoint_d = {get_key_from_feature(f):f for f in data_points}

    prediction = []

    found_n = 0
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)

        prediction_list = []
        for _text, _pid, _score in lucene_results:
            p_entry = {
                'cid': cid,
                'pid': _pid,
                'claim_text': claim_text,
                'perspective_text': _text,
                'rationale': "es score={}".format(_score),
                'score': _score
            }
            query_id = "{}_{}".format(cid, _pid)
            if query_id in datapoint_d:
                score_adjust = get_adjusted_score(datapoint_d[query_id])
                print(_score, score_adjust)
                p_entry['score'] += score_adjust
                found_n += 1
            prediction_list.append(p_entry)

        prediction_list.sort(key=lambda x:x['score'], reverse=True)
        prediction_list = prediction_list[:top_k]

        prediction.append((cid, prediction_list))
    print("{} found of {}".format(found_n, 50*len(claims)))
    return prediction


def send_bm25_query(query_id, tokens, K = 0):
    query = format_query_bm25(query_id, tokens, K)
    return send_doc_queries(get_first_disk(), 100, [query])[query_id]


def load_feature_and_split():
    print("Loading data")
    train_data = load_from_pickle("pc_train_features_binary")
    return split_7_3(train_data)


