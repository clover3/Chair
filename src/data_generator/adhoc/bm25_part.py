import random
from data_generator.adhoc.data_sampler import DataSampler
from adhoc.bm25 import BM25_2
from collections import Counter
import pickle


def sample_debiase(data_list):
    max_occurence = 2

    output = {}
    for v_list, score in data_list:
        score_grouper = int(score * 2 + 0.8)
        if score_grouper not in output:
            output[score_grouper] = []
        if len(output[score_grouper]) < max_occurence:
            output[score_grouper].append((v_list,score))

    candidate = []
    for key_score, span_list in output.items():
        for v_list, score in span_list:
            candidate.append((v_list, score))
    return candidate

def generate_data():
    ds = DataSampler.init_from_pickle("robust04")

    collection_len = 252359881
    avdl = collection_len / len(ds.collection)
    random.shuffle(ds.queries)
    window_size = 200 * 3

    def flatten_and_get_doc_id(postings_list):
        doc_ids = []
        for postings in postings_list:
            for doc_id, idx in postings:
                doc_ids.append(doc_id)
        return doc_ids

    for query in ds.queries:
        q_terms = query.split()
        postings_list = []
        for qterm in q_terms:
            postings = ds.inv_index[qterm]
            postings_list.append(postings)
        print("Query :", query)
        doc_id_list = flatten_and_get_doc_id(postings_list)
        print("Docs : {}".format(len(doc_id_list)))
        if len(doc_id_list) < 100000:
            continue
        candidates = []
        for doc_id in doc_id_list[:100]:
            raw_document = ds.collection[doc_id]
            loc_ptr = window_size
            v_list = []
            while loc_ptr < len(raw_document):
                text_span = raw_document[loc_ptr:loc_ptr + window_size]
                tf_dict = Counter(text_span.lower().split())
                dl = sum(tf_dict.values())
                span_v = []
                for q_term in q_terms:
                    tf = tf_dict[q_term]
                    df = ds.idf.df[q_term]
                    score = BM25_2(tf, df, N=len(ds.collection), dl=dl, avdl=avdl)
                    span_v.append(score)
                loc_ptr += window_size
                v_list.append(span_v)

            tf_dict = Counter(raw_document.lower().split())
            dl = sum(tf_dict.values())

            g_score = 0
            for q_term in q_terms:
                tf = tf_dict[q_term]
                df = ds.idf.df[q_term]
                g_score += BM25_2(tf, df, N=len(ds.collection), dl=dl, avdl=avdl)

            candidates.append((v_list, g_score))

        sample_space = sample_debiase(candidates)
        ## TODO select samples to train pairwise rank

        n = max(len(candidates), 1000)
        for i in range(n):
            c1, c2 = random.sample(sample_space, 2)
            v1, score1 = c1
            v2, score2 = c2
            if score1 < score2:
                yield (v1, v2)
            else:
                yield (v2, v1)



if __name__ == '__main__':
    items = []
    for pair in generate_data():
        items.append(pair)
        if len(items) > 100000:
            break

    pickle.dump(items, open("items.pickle", "wb"))


