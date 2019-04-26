import numpy as np
from nltk.tokenize import wordpunct_tokenize
import math
import collections
from collections import Counter
from models.classic.stopword import load_stopwords



class LMClassifierEx:
    def __init__(self, tokenizer=wordpunct_tokenize, stemmer=None):
        self.tokenizer = tokenizer
        self.alpha_list = None

        self.C_ctf = []
        self.C = []
        self.BG_ctf = None
        self.BG = None
        self.smoothing = 0.1
        self.stemmer = stemmer
        self.fulltext = False
        self.supervised = False
        self.n_lm = 2


    def build(self, lm_docs_list, bg_tf, bg_ctf):
        self.n_lm = len(lm_docs_list)

        stopwords = load_stopwords()

        def transform(counter):
            if self.stemmer is None:
                new_tf = counter
            else:
                new_tf = Counter()
                for key in counter:
                    source = key
                    target = self.stemmer(key)
                    new_tf[target] += counter[source]

            counter = new_tf
            new_tf = Counter()
            for key in counter:
                if len(key) <= 3 or key in stopwords:
                    pass
                else:
                    new_tf[key] = counter[key]
            return new_tf

        def remove_stopword(counter):
            new_tf = Counter()
            for key in counter:
                if len(key) < 3 or key in stopwords:
                    pass
                else:
                    new_tf[key] = counter[key]
            return new_tf

        self.BG = transform(bg_tf)
        self.BG_ctf = bg_ctf
        self.stopword = stopwords

        for lm_docs in lm_docs_list:
            c_tf = collections.Counter()
            for idx, s in enumerate(lm_docs):
                tokens = self.tokenizer(s)
                for token in tokens:
                    if token in bg_tf:
                        c_tf[token] += 1

            tf_dict = transform(c_tf)
            self.C.append(tf_dict)
            self.C_ctf.append(sum(tf_dict.values()))


    def tokenize(self, str):
        tokens = self.tokenizer(str)
        if self.stemmer:
            tokens = list([self.stemmer(t) for t in tokens])
        return tokens

    def get_tf10(self, tokens):
        counter = Counter()
        for t in tokens:
            if t not in self.stopword and len(t) > 2:
                counter[t] += 1

        return counter.most_common(10)


    def log_likelihood_base(self, tokens):
        sum_likeli = np.array([0.0 for _ in range(self.n_lm+1)])
        if self.fulltext:
            for token in set(tokens):
                s = self.term_likely(token)
                sum_likeli += s
        else:
            for token, _ in self.get_tf10(tokens):
                s = self.term_likely(token)
                sum_likeli += s

        return np.array(sum_likeli)

    def log_likelihood(self, tokens):
        list_ll = self.log_likelihood_base(tokens)
        for i in range(self.n_lm+1):
            list_ll[i] += self.alpha_list[i]
        return list_ll

    def predict(self, data):
        y = []
        for idx, s in enumerate(data):
            tokens = self.tokenize(s)
            list_ll = self.log_likelihood(tokens)

            label = list_ll[0] - max(list_ll[1:]) > 0
            y.append(label)

        return np.array(y)

    def get_score(self, doc):
        tokens = self.tokenize(doc)
        ll = self.log_likelihood(tokens)
        return ll[0] - max(ll[1:])

    def tune_alpha(self, x, y):
        vectors = []
        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            ll = self.log_likelihood_base(tokens)
            vectors.append((ll, y[idx]))

        def get_acc(alpha_list):
            p = 0
            n = 0
            for vector in vectors:
                v, y = vector
                z = alpha_list + v
                label = int(z[0] > max(z[1:]))
                if label == y:
                    p += 1
                else:
                    n += 1

            acc = p / (p + n)
            return acc

        param = Counter()
        for k in range(-500, 500, 10):
            alpha_0 = k / 100
            self.alpha_list = np.array([alpha_0] + self.n_lm * [0])
            param[alpha_0] = get_acc(self.alpha_list)

        print(param)
        alpha_0 = param.most_common(1)[0][0]

        param = Counter()
        for k in range(-500, 500, 2):
            alpha_1 = k / 100
            self.alpha_list = np.array([alpha_0, alpha_1, 0])
            param[alpha_1] = get_acc(self.alpha_list)

        print(param)
        alpha_1 = param.most_common(1)[0][0]

        param = Counter()
        for k in range(-500, 500, 2):
            alpha_0 = k / 100
            self.alpha_list = np.array([alpha_0, alpha_1, 0])
            param[alpha_0] = get_acc(self.alpha_list)

        print(param)
        alpha_0 = param.most_common(1)[0][0]

        self.alpha_list = np.array([alpha_0, alpha_1, 0])
        print(self.alpha_list)
        print("Train acc : {}".format(param.most_common(1)[0][1]))

    def term_likely(self, token):
        if token in self.stopword:
            return 0

        def count(LM, token):
            if token in LM:
                return LM[token]
            else:
                return 0


        tf_bg = count(self.BG, token)
        if tf_bg == 0:
            return 0

        P_w_BG = tf_bg / self.BG_ctf
        if P_w_BG == 0:
            return 0
        assert P_w_BG > 0
        logBG = math.log(P_w_BG)


        list_likely = []
        for i in range(self.n_lm):
            tf_c = count(self.C[i], token)
            P_w_C = tf_c / self.C_ctf[i]
            logC = math.log(P_w_C * self.smoothing + P_w_BG * (1 - self.smoothing))
            list_likely.append(logC)
            assert (math.isnan(logC) == False)

        list_likely.append(logBG)

        return list_likely
