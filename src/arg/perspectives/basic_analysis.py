from typing import List, Dict, Tuple

import nltk

from arg.perspectives import es_helper
from arg.perspectives.collection_based_classifier import predict_interface
from arg.perspectives.declaration import PerspectiveCandidate
from arg.perspectives.load import get_claim_perspective_id_dict, get_perspective_dict, load_claim_perspective_pair, \
    get_claims_from_ids, load_claim_ids_for_split
from cie.msc.tf_idf import sublinear_term_frequency, cosine_similarity, inverse_document_frequencies
from list_lib import flatten
from misc_lib import split_7_3, NamedNumber


def get_candidates(claims, balance) -> List[PerspectiveCandidate]:
    related_p_map = get_claim_perspective_id_dict()
    related_p_map = {key: flatten(value) for key, value in related_p_map.items()}
    p_map = get_perspective_dict()

    all_data_points = []
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)

        rp = related_p_map[cid]

        pid_set = list([_pid for _text, _pid, _score in lucene_results])
        data_point_list = []
        for pid in pid_set:
            p_text = p_map[pid]
            label = 1 if pid in rp else 0
            data_point = PerspectiveCandidate(label=str(label), cid=cid, pid=pid,
                                              claim_text=claim_text, p_text=p_text)
            #data_point = [str(label), str(cid), str(pid), claim_text, p_text]
            data_point_list.append(data_point)

        # If training, we balance positive and negative examples.
        if balance:
            pos_insts = list([e for e in data_point_list if e.label == "1"])
            neg_insts = list([e for e in data_point_list if e.label == "0"])
            neg_insts = neg_insts[:len(pos_insts)]
            data_point_list = pos_insts + neg_insts
        all_data_points.extend(data_point_list)

    return all_data_points


class MyIdf:
    def __init__(self):
        claim_and_perspective = load_claim_perspective_pair()
        perspective = get_perspective_dict()
        all_sents = []
        for e in claim_and_perspective:
            claim_text = e['text']
            all_sents.append(claim_text)

        for pid, text in perspective.items():
            all_sents.append(text)

        print("tokenizing {} docs".format(len(all_sents)))
        token_docs = []
        for s in all_sents:
            tokens = nltk.sent_tokenize(s)
            token_docs.append(tokens)

        print("get_idf")
        idf = inverse_document_frequencies(token_docs)
        tfidf_documents = []
        print("sublinear tf")
        for document in token_docs:
            doc_tfidf = []
            for term in idf.keys():
                tf = sublinear_term_frequency(term, document)
                doc_tfidf.append(tf * idf[term])
            tfidf_documents.append(doc_tfidf)

        self.d = {}
        for sent, tfidf_val in zip(all_sents, tfidf_documents):
            self.d[sent] = tfidf_val

    def get_score(self, d1, d2):
        return cosine_similarity(self.d[d1], self.d[d2])


def trivial_similarity(t1, t2):
    tokens1 = set(nltk.word_tokenize(t1.lower()))
    tokens2 = set(nltk.word_tokenize(t2.lower()))

    diff1 = tokens1.difference(tokens2)
    diff2 = tokens2.difference(tokens1)

    if len(diff1) < 0.2 * len(tokens1) and len(diff2) < 0.2 * len(tokens2):
        return True
    else:
        return False


# input : claims, top_k
# output : List(cid, List[dict])
def predict_by_elastic_search(claims, top_k):
    prediction = []
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
                'score': _score,
                'rationale': "es score={}".format(_score)
            }
            prediction_list.append(p_entry)

        prediction_list = prediction_list[:top_k]

        prediction.append((cid, prediction_list))
    return prediction


def predict_by_oracle_on_candidate(
                  claims,
                  top_k) -> List[Tuple[str, List[Dict]]]:
    gold: Dict[int, List[List[int]]] = get_claim_perspective_id_dict()

    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        gold_pids = gold[int(claim_id)]
        score = 0
        for p_ids in gold_pids:
            if int(p_id) in p_ids:
                score = 1

        return NamedNumber(score, "")

    r = predict_interface(claims, top_k, scorer)
    return r


def test_es():
    claim_and_perspective = load_claim_perspective_pair()
    perspective = get_perspective_dict()

    for e in claim_and_perspective:
        claim_text = e['text']
        for perspective_cluster in e['perspectives']:
            pids = perspective_cluster['pids']
            for pid in pids:
                query = claim_text + " " + perspective[pid]
                es_helper.get_perspective_from_pool(query, 50)


def load_train_data_point():
    return load_data_point('train')


def load_data_point(split):
    d_ids = list(load_claim_ids_for_split(split))
    claims = get_claims_from_ids(d_ids)
    is_train = split == "train"
    all_data_points = get_candidates(claims, is_train)
    return all_data_points


def load_data_point_50(split):
    d_ids = list(load_claim_ids_for_split(split))
    claims = get_claims_from_ids(d_ids)
    all_data_points = get_candidates(claims, False)
    return all_data_points


def load_data_point_50_train_val(split):
    d_ids = list(load_claim_ids_for_split("train"))
    train_ids, val_ids = split_7_3(d_ids)
    print(len(d_ids), len(train_ids), len(val_ids))
    if split == "train":
        sel_ids = train_ids
    elif split == "val":
        sel_ids = val_ids
    else:
        sel_ids = d_ids
        print("Taking all claims in {}".format(split))

    claims = get_claims_from_ids(sel_ids)
    all_data_points = get_candidates(claims, False)
    return all_data_points