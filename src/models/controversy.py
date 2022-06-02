import math
from multiprocessing import Pool

from krovetzstemmer import Stemmer

from data_generator.data_parser import amsterdam
from data_generator.data_parser import controversy
from list_lib import lmap, left
from misc_lib import *
from models.classic.lm_classifier import LMClassifer
from models.classic.stopword import load_stopwords
from summarization.tokenizer import *


def get_dbpedia_contrv_lm():
    print("Building LM from DBPedia's controversy ranked docs")

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

    classifier = LMClassifer(tokenizer, stemmer, fulltext=False)
    classifier.build(cont_docs_text, bg_tf, bg_ctf, )
    return classifier



def get_wiki_doc_lm(fulltext=False):
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
    classifier = LMClassifer(tokenizer, stemmer, fulltext=True)
    classifier.build2(all_docs, y)
    return classifier


def get_wiki_doc():
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

    X = []
    voca = set()
    for doc in all_docs:
        tokens = tokenizer(doc)
        voca.update(tokens)
        X.append(tokens)
    return X, y, voca


def count_word(documents):
    counter = Counter()
    for doc in documents:
        for i, token in enumerate(doc):
            if token == "PADDING":
                break
            counter[token] += 1
    return counter

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def from_pos_neg(pos_docs, neg_docs):
    stemmer = None
    stopwords = load_stopwords()
    y = list(1 for _ in pos_docs) + list(0 for _ in neg_docs)

    def transform(counter):
        if stemmer is None:
            new_tf = counter
        else:
            new_tf = Counter()
            for key in counter:
                source = key
                target = stemmer(key)
                new_tf[target] += counter[source]

        counter = new_tf
        new_tf = Counter()
        for key in counter:
            if len(key) <= 3 or key in stopwords:
                pass
            else:
                new_tf[key] = counter[key]
        return new_tf

    def count_word_parallel(documents):
        split = 30
        p = Pool(split)
        args = chunks(documents, split)
        counters = p.map(count_word, args)
        g_counter = Counter()
        for counter in counters:
            for key in counter.keys():
                g_counter[key] += counter[key]
        return g_counter

    c_counter = transform(count_word_parallel(pos_docs))
    nc_counter = transform(count_word_parallel(neg_docs))

    tokenizer = lambda x: tokenize(x, set(), False)
    classifier = LMClassifer(tokenizer, None, fulltext=True)
    classifier.build3(c_counter, nc_counter)
    return classifier


def get_guardian16_lm():
    print("Building LM from guardian16 signal")
    pos_docs, neg_docs = controversy.load_guardian16_signal()
    return from_pos_neg(pos_docs, neg_docs)

def get_guardian_selective_lm():
    print("Building LM from seletive signal")
    pos_docs, neg_docs = controversy.load_guardian_selective_signal()
    return from_pos_neg(pos_docs, neg_docs)


def get_yw_may():
    from old_projects.dispute.guardian import load_local_pickle


    stopwords = load_stopwords()
    tokenizer = lambda x: tokenize(x, stopwords, False)

    class YWMay:
        def __init__(self):
            self.stopwords = stopwords
            self.topic_info = load_local_pickle("topic_score")


        def get_tf10(self, tokens):
            counter = Counter()
            for t in tokens:
                if t not in self.stopwords and len(t) > 2:
                    counter[t] += 1

            return counter.most_common(10)

        def score(self, docs):
            def term_odd(token):
                if token not in self.topic_info:
                    return 0
                else:
                    p = self.topic_info[token]
                    if p > 0.9999 or p < 0.0001:
                        return 0
                    else:
                        return math.log(p) - math.log(1-p)

            def predict(doc):
                tokens = tokenizer(doc)
                sum_odd = 0

                top10 = left(list(self.get_tf10(tokens)))
                odd_list = lmap(term_odd, tokens)
                result = sum(odd_list)
                return result
            return lmap(predict, docs)

    return YWMay()
