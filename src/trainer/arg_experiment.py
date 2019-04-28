from data_generator.argmining import ukp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier

from task.metrics import eval_3label, eval_2label


from models.classic.stopword import load_stopwords
from summarization import tokenizer
from summarization.text_rank import TextRank
from cie import claim_gen
from misc_lib import flatten, average, tprint
from cie.arg import kl

import numpy as np
from collections import Counter
import math



class ArgExperiment:
    def __init__(self):
        pass


    def train_lr_3way(self):
        topic = ukp.all_topics[0]

        data_loader = ukp.DataLoader(topic)
        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        train_data = data_loader.get_train_data()
        dev_data = data_loader.get_dev_data()

        train_X, train_y = zip(*train_data)
        dev_X, dev_y = zip(*dev_data)
        feature = CountVectorizer()
        train_X_v = feature.fit_transform(train_X)
        dev_X_v = feature.transform(dev_X)

        kl_regression = kl.KLPredictor(dev_X)

        lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
        lr.fit(train_X_v, train_y)

        train_pred = lr.predict(train_X_v)
        dev_pred = lr.predict(dev_X_v)

        def print_eval(pred_y, gold_y):
            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])

        print("Train")
        print_eval(train_pred, train_y)
        print("Dev")
        print_eval(dev_pred, dev_y)

    def train_lr_2way(self):
        topic = ukp.all_topics[0]

        data_loader = ukp.DataLoader(topic, False)
        idx_arg = 1

        train_data = data_loader.get_train_data()
        dev_data = data_loader.get_dev_data()

        train_X, train_y = zip(*train_data)
        dev_X, dev_y = zip(*dev_data)
        feature = CountVectorizer()
        train_X_v = feature.fit_transform(train_X)
        dev_X_v = feature.transform(dev_X)

        portion_true = sum(train_y) / len(train_y)

        kl_regression = kl.KLPredictor(dev_X)
        lr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        lr.fit(train_X_v, train_y)

        train_pred = lr.predict(train_X_v)
        dev_pred = lr.predict(dev_X_v)
        kl_regression.tune(dev_X, portion_true)
        dev_kl_pred = list([kl_regression.predict(sent) for sent in dev_X])


        def print_eval(pred_y, gold_y):
            all_result = eval_2label(pred_y, gold_y)
            arg_result = all_result[idx_arg]
            f1 = sum([result['f1'] for result in all_result]) / 2
            print("F1", f1)
            print("P_arg", arg_result['precision'])
            print("R_arg", arg_result['recall'])

        print("Train")
        print_eval(train_pred, train_y)
        print("Dev")
        print_eval(dev_pred, dev_y)

        print("KL - Dev")
        print_eval(dev_kl_pred, dev_y)

    def tf_stat(self):
        topic = ukp.all_topics[0]
        data_loader = ukp.DataLoader(topic)
        stopwords = load_stopwords()

        def tokenize(x):
            return tokenizer.tokenize(x, stopwords)

        for topic in ukp.all_topics:
            print("-----------")
            print(topic)
            entries = data_loader.all_data[topic]
            token_sents = list([tokenize(e['sentence']) for e in entries if e['set'] == 'train'])
            tf_dict = Counter(flatten(token_sents))
            for word, tf in tf_dict.most_common(30):
                print(word, tf)

    def divergence(self):
        # Compare Arg vs Non-Arg
        topic = ukp.all_topics[0]
        data_loader = ukp.DataLoader(topic)
        stopwords = load_stopwords()

        def tokenize(x):
            return tokenizer.tokenize(x, stopwords)


        def is_argument(entry):
            return entry['annotation'] == "Argument_for" or entry['annotation'] == "Argument_against"

        for topic in ukp.all_topics:
            print("-----------")
            print(topic)
            entries = data_loader.all_data[topic]
            token_sents = list([tokenize(e['sentence']) for e in entries if e['set'] == 'train'])
            topic_tf = Counter(flatten(token_sents))

            arg_div = []
            narg_div = []
            for e in entries:
                sent_tf = Counter(tokenize(e['sentence']))
                div = kl.kl_divergence(sent_tf, topic_tf)
                assert not math.isnan(div)

                if e['set'] == 'train' and is_argument(e):
                    arg_div.append(div)
                elif e['set'] == 'train':
                    narg_div.append(div)


            print("Arg KL mean : " , average(arg_div))
            print("Non-Arg KL mean : ", average(narg_div))


    def divergence_lr(self):
        f1_list = []
        for dev_topic in ukp.all_topics:
            print(dev_topic)
            data_loader = ukp.DataLoader(dev_topic)
            idx_for = data_loader.labels.index("Argument_for")
            idx_against = data_loader.labels.index("Argument_against")

            train_data = data_loader.get_train_data()
            dev_data = data_loader.get_dev_data()

            train_X, train_y = zip(*train_data)
            dev_X, dev_y = zip(*dev_data)
            feature = CountVectorizer()
            train_X_v = feature.fit_transform(train_X)

            stopwords = load_stopwords()
            def tokenize(x):
                return tokenizer.tokenize(x, stopwords)

            data_idx = 0
            for topic in ukp.all_topics:
                if topic == dev_topic:
                    continue
                entries = data_loader.all_data[topic]
                token_sents = list([tokenize(e['sentence']) for e in entries if e['set'] == 'train'])
                topic_tf = Counter(flatten(token_sents))

                for e in entries:
                    if e['set'] == 'train':
                        sent_tf = Counter(tokenize(e['sentence']))
                        div = kl.kl_divergence(sent_tf, topic_tf)
                        assert not math.isnan(div)
                        train_X_v[data_idx,-1] = div
                        data_idx += 1

            assert data_idx == len(train_X)

            classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
            #classifier = LinearSVC()
            classifier = MLPClassifier()


            classifier.fit(train_X_v, train_y)

            dev_X_v = feature.transform(dev_X)

            token_sents = list([tokenize(e['sentence']) for e in data_loader.all_data[dev_topic] if e['set'] == 'val'])
            topic_tf = Counter(flatten(token_sents))
            data_idx = 0
            for e in data_loader.all_data[dev_topic]:
                if e['set'] == 'val':
                    sent_tf = Counter(tokenize(e['sentence']))
                    div = kl.kl_divergence(sent_tf, topic_tf)
                    train_X_v[data_idx, -1] = div
                    data_idx += 1

            assert data_idx == len(dev_X)
            train_pred = classifier.predict(train_X_v)
            dev_pred = classifier.predict(dev_X_v)

            def print_eval(pred_y, gold_y):
                all_result = eval_3label(pred_y, gold_y)
                for_result = all_result[idx_for]
                against_result = all_result[idx_against]
                f1 = sum([result['f1'] for result in all_result]) / 3
                print("F1", f1)
                print("P_arg+", for_result['precision'])
                print("R_arg+", for_result['recall'])
                print("P_arg-", against_result['precision'])
                print("R_arg-", against_result['recall'])
                return f1

            #print("Train")
            #print_eval(train_pred, train_y)

            f1 = print_eval(dev_pred, dev_y)
            f1_list.append(f1)
        average(f1_list)


    def summarize(self):
        topic = ukp.all_topics[0]
        data_loader = ukp.DataLoader(topic)
        stopwords = load_stopwords()

        def tokenize(x):
            return tokenizer.tokenize(x, stopwords)

        def sent_score(token_sent, bow_score):
            score = 0
            factor = 1
            for t in token_sent:
                score += bow_score[t] * factor
                factor *= 0.5
            return score



        def is_argument(entry):
            return entry['annotation'] == "Argument_for" or entry['annotation'] == "Argument_against"

        for topic in ukp.all_topics:
            entries = data_loader.all_data[topic]
            raw_sents = list([e['sentence'] for e in entries if e['set'] == 'train'])
            token_sents = list(map(tokenize, raw_sents))
            tprint("Runing TextRank")
            text_rank = TextRank(token_sents)
            tr_score = Counter(text_rank.run(flatten(token_sents)))
            tprint("claim_gen.generate")

            raw_sents.sort(key=lambda x: sent_score(tokenize(x), tr_score), reverse=True)
            for i in range(10):
                print(raw_sents[i])

            #claim_gen.generate(raw_sents, tr_score)

