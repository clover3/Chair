import os

import nltk

from arg.perspectives import es_helper
from arg.perspectives.evaluate import evaluate
from arg.perspectives.load import get_claim_perspective_id_dict, get_perspective_dict, get_claims_from_ids, \
    load_claim_perspective_pair, load_train_claim_ids
from cie.msc.tf_idf import sublinear_term_frequency, cosine_similarity, inverse_document_frequencies
from cpath import output_path
from misc_lib import flatten
from tlm.retrieve_lm.galago_query_maker import get_query_entry_bm25_anseri, save_queries_to_file, clean_query


def get_candidates(claims, is_train=True):
    related_p_map = get_claim_perspective_id_dict()
    related_p_map = {key: flatten(value) for key, value in related_p_map.items()}
    p_map = get_perspective_dict()

    all_data_points = []
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)

        rp = related_p_map[cid]

        pid_set = [_pid for _text, _pid, _score in lucene_results]
        data_point_list = []
        for pid in pid_set:
            p_text = p_map[pid]
            label = 1 if pid in rp else 0
            data_point = [str(label), str(cid), str(pid), claim_text, p_text]
            data_point_list.append(data_point)

        # If training, we balance positive and negative examples.
        if is_train:
            pos_insts = list([e for e in data_point_list if e[0] == "1"])
            neg_insts = list([e for e in data_point_list if e[0] == "0"])
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


def predict_by_elastic_search(claims, top_k):
    prediction = []
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)

        pid_set = []
        for _text, _pid, _score in lucene_results:
            pid_set.append(_pid)

        pid_set = pid_set[:top_k]

        prediction.append((cid, pid_set))
    return prediction


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


def run_baseline():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    pred = predict_by_elastic_search(claims, 30)
    print(evaluate(pred))


def write_claim_as_query():
    d_ids = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    queries = []
    for c in claims:
        cid = c["cId"]
        claim_text = c["text"]
        tokens = claim_text.split()
        query_text = clean_query(tokens)
        print(query_text)
        q_entry = get_query_entry_bm25_anseri(cid, query_text)
        queries.append(q_entry)

    out_path = os.path.join(output_path, "perspective_dev_claim_query.json")
    save_queries_to_file(queries, out_path)


if __name__ == "__main__":
    run_baseline()
