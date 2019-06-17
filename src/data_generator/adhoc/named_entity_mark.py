import spacy
import en_core_web_sm
from config.input_path import robust_path
from data_generator.data_parser.trec import *
import random
from data_generator.adhoc.data_sampler import DataSampler

nlp = en_core_web_sm.load()

def mark_NE_robust():
    tprint("Loading collection")
    robust_collection = load_robust(robust_path, False)


    NE_index = defaultdict(list)
    tprint("Reading collections")

    keys = list(robust_collection.keys())
    random.shuffle(keys)

    todo = keys[:3600]

    ticker = TimeEstimator(len(todo))
    for doc_id in todo:
        doc = robust_collection[doc_id]
        parsed_doc = nlp(doc)
        for X in parsed_doc.ents:
            #pprint((X.text, X.label_))
            X = str(X.text), str(X.label_)
            NE_index[X].append(doc_id)

        ticker.tick()

    save_to_pickle(NE_index, "ne_indx")


def save_inv_index():
    data_sampler = load_from_pickle("robust04")
    inv_index = data_sampler.inv_index
    save_to_pickle(inv_index, "robust04_inv_index")


def inpsect_ne_pair():
    ne_idx = load_from_pickle("ne_indx")
    #inv_indx = load_from_pickle("robust04_inv_index")

    doc2ne = defaultdict(set)



    def retrieve_sent(ne1, ne2):
        NotImplemented


    ne_pair_count = Counter()
    ne_neighbor = defaultdict(list)

    for ne, postings in ne_idx.items():
        ne_text, ne_label = ne
        for doc_id in postings:
            doc2ne[doc_id].add(ne)

    for ne, postings in ne_idx.items():
        for doc_id in postings:
            for ne2 in doc2ne[doc_id]:
                if ne != ne2:
                    ne_neighbor[ne].append(ne2)


    for ne in ne_neighbor:
        neighbors = ne_neighbor[ne]
        n_count = Counter(neighbors)
        for ne2, cnt in n_count.most_common(3):
            if cnt > 1:
                print(ne , ne2 , cnt)




    return
    appeared_docs = NotImplemented

    ne_pairs = []
    for doc_id in NotImplemented:
        ne_list = doc2ne[doc_id]
        for ne2 in ne_list:
            ne_pairs.append((ne, ne2))

    common_ne = NotImplemented
    for ne1, ne2 in common_ne:
        sent = retrieve_sent(ne1, ne2)




if __name__ == '__main__':
    inpsect_ne_pair()

