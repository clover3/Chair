import spacy
import en_core_web_sm
from config.input_path import robust_path
from data_generator.data_parser.trec import *
import random

nlp = en_core_web_sm.load()

def mark_NE_robust():
    tprint("Loading collection")
    robust_collection = load_robust(robust_path)


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
            NE_index[X].append(doc_id)

        ticker.tick()

    save_to_pickle(NE_index, "ne_indx")



if __name__ == '__main__':
    mark_NE_robust()

