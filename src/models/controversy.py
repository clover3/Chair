from data_generator.data_parser import controversy, load_protest
from data_generator.data_parser import amsterdam
from summarization.tokenizer import *
from models.classic.lm_classifier import LMClassifer

from krovetzstemmer import Stemmer


def get_dbpedia_contrv_lm():
    stemmer = Stemmer()
    cont_docs = controversy.load_pseudo_controversy_docs("dbpedia")[:7500]
    print("Using {} docs".format(len(cont_docs)))
    tokenizer = lambda x: tokenize(x, set(), False)
    assert cont_docs[0][0] == 1

    print("Loading collection stats")
    bg_ctf, bg_tf = controversy.load_tf("tf_dump_100.txt")
    bg_ctf = sum(bg_tf.values())
    cont_docs_text = list([x[2] for x in cont_docs])
    print("Building LM classifier ")

    classifier = LMClassifer(tokenizer, stemmer)
    classifier.build(cont_docs_text, bg_tf, bg_ctf, )
    return classifier



def get_wiki_doc_lm():
    print("Building LM from wikipedia controversy list")
    train_data = amsterdam.get_train_data(separate=True)
    pos_entries, neg_entries = train_data
    stemmer = Stemmer()

    def doc_rep(entry):
        return entry["title"] + "\t" + entry["content"]


    pos_docs = list(map(doc_rep, pos_entries))
    neg_docs = list(map(doc_rep, neg_entries))

    y = list(1 for _ in pos_docs) + list(0 for _ in neg_docs)
    all_docs = pos_docs + neg_docs

    tokenizer = lambda x: tokenize(x, set(), False)
    classifier = LMClassifer(tokenizer, stemmer)
    classifier.build2(all_docs, y)
    return classifier

