import numpy as np
from nltk.tokenize import wordpunct_tokenize
import math
import collections
from collections import Counter
from models.classic.stopword import load_stopwords

class LMClassifer:
    def __init__(self, tokenizer=wordpunct_tokenize, stemmer=None):
        self.tokenizer = tokenizer
        self.opt_alpha = None

        self.C_ctf = None
        self.C = None
        self.BG_ctf = None
        self.BG = None
        self.smoothing = 0.1
        self.stemmer = stemmer
        self.fulltext = False
        self.supervised = False

    def build(self, c_docs, bg_tf, bg_ctf):
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

        c_tf = collections.Counter()
        for idx, s in enumerate(c_docs):
            tokens = self.tokenizer(s)
            for token in tokens:
                if token in bg_tf:
                    c_tf[token] += 1

        self.C = transform(c_tf)
        self.C_ctf = sum(self.C.values())


    def tokenize(self, str):
        tokens = self.tokenizer(str)
        if self.stemmer:
            tokens = list([self.stemmer(t) for t in tokens])
        return tokens


    # x : list of str
    # y : list of binary label
    def build2(self, x, y):
        stopwords = load_stopwords()
        self.stopword = stopwords
        self.supervised = True

        self.NC = collections.Counter()
        self.C = collections.Counter()
        def update(counter, tokens):
            for token in tokens:
                counter[token] += 1

        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            if y[idx] == 0:
                update(self.NC, tokens)
            elif y[idx] == 1:
                update(self.C, tokens)

        self.NC_ctf = sum(self.NC.values())
        self.C_ctf = sum(self.C.values())

        vectors = []
        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            odd = self.log_odd_binary(tokens)
            vectors.append((odd, y[idx]))
        vectors.sort(key=lambda x:x[0], reverse=True)

        total = len(vectors)
        p =  np.count_nonzero(y)
        fp = 0
        max_acc = 0
        self.opt_alpha = 0
        for idx, (odd, label) in enumerate(vectors):
            alpha = odd - 1e-8
            if label == 0:
                fp += 1

            tp = (idx+1) - fp
            fn = p - tp
            tn = total - (idx+1) - fn
            acc = (tp + tn) / (total)
            if acc > max_acc:
                self.opt_alpha = alpha
                max_acc = acc

        print("Train acc : {}".format(max_acc))

    def tune_alpha(self, x, y):
        vectors = []
        for idx, s in enumerate(x):
            tokens = self.tokenize(s)
            odd = self.log_odd(tokens)
            vectors.append((odd, y[idx]))
        vectors.sort(key=lambda x:x[0], reverse=True)

        total = len(vectors)
        p =  np.count_nonzero(y)
        fp = 0
        max_acc = 0
        self.opt_alpha = 0
        for idx, (odd, label) in enumerate(vectors):
            alpha = odd - 1e-8
            if label == 0:
                fp += 1

            tp = (idx+1) - fp
            fn = p - tp
            tn = total - (idx+1) - fn
            acc = (tp + tn) / (total)
            if acc > max_acc:
                self.opt_alpha = alpha
                max_acc = acc

        print("Train acc : {}".format(max_acc))


    def predict(self, data):
        y = []
        for idx, s in enumerate(data):
            tokens = self.tokenize(s)
            odd = self.log_odd(tokens)
            y.append(int(odd > self.opt_alpha))

        return np.array(y)

    def log_odd_text(self, text):
        if not self.supervised:
            return self.log_odd(self.tokenize(text))
        else:
            return self.log_odd_binary(self.tokenizer(text))

    def term_odd(self, token):
        if token in self.stopword:
            return 0

        def count(LM, token):
            if token in LM:
                return LM[token]
            else:
                return 0

        tf_c = count(self.C, token)
        tf_bg = count(self.BG, token)
        if tf_c == 0 and tf_bg == 0:
            return 0
        P_w_C = tf_c / self.C_ctf
        P_w_BG = tf_bg / self.BG_ctf
        if P_w_BG == 0 :
            return 0

        assert P_w_BG > 0
        logC = math.log(P_w_C * self.smoothing + P_w_BG * (1 - self.smoothing))
        logNC = math.log(P_w_BG)
        assert (math.isnan(logC) == False)
        assert (math.isnan(logNC) == False)
        return logC - logNC

    def log_odd(self, tokens):
        sum_odd = 0
        if self.fulltext:
            for token in set(tokens):
                s = self.term_odd(token)
                sum_odd += s
        else:
            for token, _ in self.get_tf10(tokens):
                s = self.term_odd(token)
                sum_odd += s

        return sum_odd


    def get_tf10(self, tokens):
        counter = Counter()
        for t in tokens:
            if t not in self.stopword and len(t) > 2:
                counter[t] += 1

        return counter.most_common(10)

    def log_odd_binary(self, tokens):
        smoothing = 0.1

        def count(LM, token):
            if token in LM:
                return LM[token]
            else:
                return 0

        def per_token_odd(token):
            tf_c = count(self.C, token)
            tf_nc = count(self.NC, token)
            if tf_c == 0 and tf_nc == 0:
                return 0
            P_w_C = tf_c / self.C_ctf
            P_w_NC = tf_nc / self.NC_ctf
            P_w_BG = (tf_c + tf_nc) / (self.NC_ctf + self.C_ctf)
            logC = math.log(P_w_C * smoothing + P_w_BG * (1 - smoothing))
            logNC = math.log(P_w_NC * smoothing + P_w_BG * (1 - smoothing))
            assert (math.isnan(logC) == False)
            assert (math.isnan(logNC) == False)
            return logC - logNC

        sum_odd = 0
        if self.fulltext:
            for token in set(tokens):
                s = per_token_odd(token)
                sum_odd += s
        else:
            for token, _ in self.get_tf10(tokens):
                s = per_token_odd(token)
                sum_odd += s

        return sum_odd