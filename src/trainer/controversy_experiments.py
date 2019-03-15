from data_generator.data_parser import controversy
from summarization.tokenizer import *
from summarization.text_rank import TextRank
from models.classic.lm_classifier import LMClassifer
from evaluation import compute_auc, compute_pr_auc
from krovetzstemmer import Stemmer
from nltk.tokenize import wordpunct_tokenize

from collections import Counter


class ControversyExperiment:
    def __init__(self):
        NotImplemented


    def view_docs(self):
        docs = controversy.load_docs()
        for filename, doc in docs:
            print(doc)


    def view_keyword(self):
        docs = controversy.load_docs()
        token_docs = dict()
        for filename, doc in docs:
            token_docs[filename] = tokenize(doc, set())

        labels = controversy.load_label()

        tr = TextRank(token_docs.values())
        for key in token_docs:
            scores = Counter(tr.run(token_docs[key]))
            print("--------", labels[key])
            print(key)
            for key, value in scores.most_common(10):
                print(key, value)


    def lm_baseline(self):
        stemmer = Stemmer()
        docs = controversy.load_docs()
        cont_docs = controversy.load_pseudo_controversy_docs("dbpedia")[:7500]
        print("Using {} docs".format(len(cont_docs)))
        tokenizer = lambda x: tokenize(x, set(), False)
        assert cont_docs[0][0] == 1


        print("Loading collection stats")
        bg_ctf, bg_tf = controversy.load_tf("tf_dump_100.txt")
        labels = controversy.load_label()
        bg_ctf = sum(bg_tf.values())
        cont_docs_text = list([x[2] for x in cont_docs])
        print("Building classifier ")

        classifier = LMClassifer(tokenizer, stemmer)
        classifier.build(cont_docs_text, bg_tf, bg_ctf,)


        classifier.smoothing = 0.05
        for term in ["dance", 'copyright']:
            print(term)
            print("tf", classifier.BG[term])
            print("tf|c", classifier.C[term])
            print("odd", classifier.term_odd(term))

        print("bg cf", bg_ctf)
        print("ctf|c", classifier.C_ctf)

        y_scores = []
        y_list = []
        for name, doc in docs:
            s = classifier.log_odd_text(doc)
            y_scores.append(s)
            y_list.append(labels[name])

            tf10 = []
            for t, _ in classifier.get_tf10(tokenizer(doc)):
                tf10.append(t)

            print(name)
            print(tf10)

            if name == "clueweb09-en0002-98-06376":
                print(s)
                q_terms = []
                for t in set(tokenizer(doc)):
                    if len(t) > 2 and t not in classifier.stopword:
                        q_terms.append(t)
                q_terms.sort()
                print(q_terms)
                print(len(q_terms))

        print("AUC :", compute_auc(y_scores, y_list))
        print("PR AUC :", compute_pr_auc(y_scores, y_list))

