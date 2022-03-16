import pickle

import galagos.basic

import cpath
import explain.bert_components.load_nli_dev
from evaluation import compute_auc, compute_pr_auc, AP
from models.classic.lm_ex import LMClassifierEx
from models.controversy import *
from summarization.text_rank import TextRank


class ControversyExperiment:
    def __init__(self):
        NotImplemented


    def view_docs(self):
        docs = controversy.load_clue303_docs()
        for filename, doc in docs:
            print(doc)


    def view_keyword(self):
        docs = controversy.load_clue303_docs()
        token_docs = dict()
        for filename, doc in docs:
            token_docs[filename] = tokenize(doc, set())

        labels = controversy.load_clue303_label()

        tr = TextRank(token_docs.values())
        for key in token_docs:
            scores = Counter(tr.run(token_docs[key]))
            print("--------", labels[key])
            print(key)
            for key, value in scores.most_common(10):
                print(key, value)


    def get_tf_10(self):
        docs = controversy.load_clue303_docs()
        stopwords = load_stopwords()
        tokenizer = lambda x: tokenize(x, stopwords, False)

        dev_size= int(len(docs) * 0.2)

        result = []
        for name, doc in docs:
            tokens = tokenizer(doc)
            counter = Counter()
            for t in tokens:
                if t not in stopwords and len(t) > 2:
                    counter[t] += 1

            terms = left(list(counter.most_common(10)))
            print(name)
            print(terms)
            result.append((name, terms))

        pickle.dump(result, open("tf10_all.pickle", "wb"))






    def lm_baseline(self):
        stemmer = Stemmer()
        docs = controversy.load_clue303_docs()
        cont_docs = controversy.load_pseudo_controversy_docs("dbpedia")[:7500]
        print("Using {} docs".format(len(cont_docs)))
        tokenizer = lambda x: tokenize(x, set(), False)
        assert cont_docs[0][0] == 1


        print("Loading collection stats")
        bg_ctf, bg_tf = controversy.load_tf("tf_dump_100.txt")
        labels = controversy.load_clue303_label()
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
        train_X, train_Y = explain.bert_components.load_nli_dev.load_data("train")
        dev_X, dev_Y = explain.bert_components.load_nli_dev.load_data("dev")

        stemmer = Stemmer()
        dir_path = os.path.join(cpath.data_path, "protest", "pseudo_docs", "dbpedia")
        tf_path = os.path.join(cpath.data_path, "protest", "pseudo_docs", "tf_dump_100.txt")
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
        bg_ctf, bg_tf = galagos.basic.load_tf(tf_path)
        print("Using {} docs".format(len(cont_docs)))
        assert cont_docs[0][0] == 1

        print("Loading collection stats")
        bg_ctf = sum(bg_tf.values())
        cont_docs_text = list([x[2] for x in cont_docs])
        print("Building classifier ")

        classifier = LMClassifer(tokenizer, stemmer)
        classifier.build(cont_docs_text, bg_tf, bg_ctf,)

        def get_ap(y_rank):
            y_rank.sort(key=lambda x:x[1], reverse=True)
            return AP(left(y_rank), dev_Y)


        print(len(dev_Y))
        print(sum(dev_Y.values()))

        """
        param = Counter()
        for sm in range(1,9):
            classifier.smoothing = sm / 100
            y_rank_method = []
            for name, doc in dev_X:
                s = classifier.log_odd_text(doc)
                y_rank_method.append((name, s))
            param[sm] = get_ap(y_rank_method)
        print(param)
        """
        classifier.smoothing = 0.01
        y_rank_method = []
        y_rank_rand = []
        y_rank_sup = []
        y_rank_hand = []
        y_list = []
        for name, doc in dev_X:
            s = classifier.log_odd_text(doc)
            s2 = c2.log_odd_text(doc)
            y_rank_method.append((name, s))
            y_rank_rand.append((name, random.random()))
            y_rank_sup.append((name, s2))
            y_list.append(dev_Y[name])

            s3 = 0
            if "protest" in doc:
                s3 += 0.3
            if "police" in doc:
                s3 += 0.1
            #if "murder" in doc or "robber" in doc:
            #    s3 += -0.1
            y_rank_hand.append((name, s3))
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

        print("AP(LM) :", get_ap(y_rank_method))
        print("AP(Random) :", get_ap(y_rank_rand))
        print("AP(Sup) :", get_ap(y_rank_sup))
        print("AP(Hand) :", get_ap(y_rank_hand))


        mid = int(len(y_rank_method) / 2)
        print("Wiki", y_rank_method[0])
        print("Wiki",  y_rank_method[mid])
        print("Sup", y_rank_sup[0])
        print("Sup", y_rank_sup[mid])


    
    def lm_protext_ex(self):
        train_X, train_Y = explain.bert_components.load_nli_dev.load_data("train")
        dev_X, dev_Y = explain.bert_components.load_nli_dev.load_data("dev")

        stemmer = Stemmer()
        dir_protest = os.path.join(cpath.data_path, "protest", "pseudo_docs", "dbpedia")
        dir_crime = os.path.join(cpath.data_path, "protest", "crime_docs")
        tf_path = os.path.join(cpath.data_path, "protest", "pseudo_docs", "tf_dump_100.txt")
        tokenizer = lambda x: tokenize(x, set(), False)

        n_docs = 3000
        protest_docs = controversy.load_dir_docs(dir_protest)[:n_docs]
        protest_docs = list([x[2] for x in protest_docs])

        crime_docs = controversy.load_dir_docs(dir_crime)[:1000]
        crime_docs = list([x[2] for x in crime_docs])
        bg_ctf, bg_tf = galagos.basic.load_tf(tf_path)
        print("Using {} docs".format(len(protest_docs)))

        classifier = LMClassifierEx(tokenizer, stemmer)
        classifier.build([protest_docs, crime_docs], bg_tf, bg_ctf)
        classifier.smoothing = 0.01

        x_list = list([x[1] for x in train_X])
        y_list = list([train_Y[x[0]] for x in train_X])

        classifier.fulltext = True
        def get_ap(y_rank):
            y_rank.sort(key=lambda x: x[1], reverse=True)
            return AP(left(y_rank), dev_Y)

        classifier.tune_alpha(x_list, y_list)

        y_rank_method = []
        for name, doc in dev_X:
            s = classifier.get_score(doc)
            y_rank_method.append((name, s))
        print("AP(LM_ex) :", get_ap(y_rank_method))

        classifier.alpha_list = [0,-9999,0]
        y_rank_method = []
        for name, doc in dev_X:
            s = classifier.get_score(doc)
            y_rank_method.append((name, s))
        print("AP(LM_ex) before tune:", get_ap(y_rank_method))

