from data_generator.data_parser import controversy, load_protest
from summarization.tokenizer import *
from summarization.text_rank import TextRank
from models.classic.lm_classifier import LMClassifer
from evaluation import compute_auc, compute_pr_auc, compute_acc, AP
from krovetzstemmer import Stemmer
from nltk.tokenize import wordpunct_tokenize
from misc_lib import left
import random
import os
import path

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

        y_scores = []
        y_list = []
        top_k = int(len(docs) * 0.3)
        bottom_k = int(len(docs) * 0.7)
        doc_dict = dict()

        ranked_list = []
        for name, doc in docs:
            doc_dict[name] = doc
            s = classifier.log_odd_text(doc)
            y_scores.append(s)
            y_list.append(labels[name])
            ranked_list.append((name, s))

            tf10 = []
            for t, _ in classifier.get_tf10(classifier.tokenize(doc)):
                tf10.append(t)

        ranked_list.sort(key=lambda x:x[1], reverse=True)
        cnt = 0
        for name, score in ranked_list:
            cnt += 1
            fp = cnt < top_k and labels[name] == 0
            fn = cnt > bottom_k and labels[name] == 1
            if fp or fn:
                print(name)
                doc = doc_dict[name]
                print(doc[:300])
                print(labels[name])
                print("Score : ", score)
                tf10 = []
                for t, _ in classifier.get_tf10(classifier.tokenize(doc)):
                    tf10.append(t)
                for term in tf10:
                    print("{}\t{}".format(term, classifier.term_odd(term)))


        print("AUC :", compute_auc(y_scores, y_list))
        print("PR AUC :", compute_pr_auc(y_scores, y_list))




    def lm_protest_baseline(self):
        train_X, train_Y = load_protest.load_data("train")
        dev_X, dev_Y = load_protest.load_data("dev")

        stemmer = Stemmer()
        dir_path = os.path.join(path.data_path, "protest", "pseudo_docs", "dbpedia")
        tf_path = os.path.join(path.data_path, "protest", "pseudo_docs", "tf_dump_100.txt")
        tokenizer = lambda x: tokenize(x, set(), False)

        c2 = LMClassifer(tokenizer, stemmer)

        x_list = []
        y_list = []
        for name, doc in train_X:
            y_list.append(train_Y[name])
            x_list.append(doc)
        c2.build2(x_list, y_list)
        n_docs = 3000
        cont_docs = controversy.load_dir_docs(dir_path)[:n_docs]
        bg_ctf, bg_tf = controversy.load_tf_inner(tf_path)
        print("Using {} docs".format(len(cont_docs)))
        assert cont_docs[0][0] == 1

        print("Loading collection stats")
        bg_ctf = sum(bg_tf.values())
        cont_docs_text = list([x[2] for x in cont_docs])
        print("Building classifier ")

        classifier = LMClassifer(tokenizer, stemmer)
        classifier.build(cont_docs_text, bg_tf, bg_ctf,)

        classifier.smoothing = 0.01

        print(len(dev_Y))
        print(sum(dev_Y.values()))
        y_rank_method = []
        y_rank_rand = []
        y_rank_sup = []
        y_list = []
        for name, doc in dev_X:
            s = classifier.log_odd_text(doc)
            s2 = c2.log_odd_text(doc)
            y_rank_method.append((name, s))
            y_rank_rand.append((name, random.random()))
            y_rank_sup.append((name, s2))
            y_list.append(dev_Y[name])

            tf10 = []
            for t, _ in classifier.get_tf10(classifier.tokenize(doc)):
                tf10.append(t)

            print(name)
            print(doc[:30])
            print(dev_Y[name])
            print("Score : ", s)
            for term in tf10:
                print("{}\t{}".format(term, classifier.term_odd(term)))
            print()
            print("Score(sup):", s2)
            for term in tf10:
                print("{}\t{}".format(term, c2.log_odd_binary([term])))
            print()

        def get_ap(y_rank):
            y_rank.sort(key=lambda x:x[1], reverse=True)
            return AP(left(y_rank), dev_Y)

        print("AP(LM) :", get_ap(y_rank_method))
        print("AP(Random) :", get_ap(y_rank_rand))
        print("AP(Sup) :", get_ap(y_rank_sup))