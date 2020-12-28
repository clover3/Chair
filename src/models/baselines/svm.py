from collections import Counter
from typing import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


def train_svm_and_test(feature_extractor, train_x, train_y, test_x):
    X_train_counts = feature_extractor.fit_transform(train_x)
    x_test_count = feature_extractor.transform(test_x)

    svclassifier = LinearSVC()
    svclassifier.fit(X_train_counts, train_y)

    train_pred = svclassifier.predict(X_train_counts)

    return svclassifier.predict(x_test_count)


def ngram_rep(g):
    return ' '.join(g)


def ngram_rep2(g):
    ext = ["{}({})".format(t, ord(t)) for t in g]
    return ' '.join(ext)


def find_ngrams(input_list, n):
    ngrams: Iterable[Iterable] = zip(*[input_list[i:] for i in range(n)])
    return [ngram_rep2(g) for g in ngrams]


def find_ngram_range(sent, n_range):
    r = []
    for n in n_range:
        r += find_ngrams(sent, n)
    return r


class NGramFeature:
    def __init__(self, use_char_ngram=True, ngram_max=3):
        self.word_feature = None
        self.use_char_ngram = use_char_ngram
        self.ngram_max = ngram_max

    def transform(self, X):
        x_word_feature = self.word_feature.transform(X)

        assert x_word_feature.shape[0] == len(X)

        num_inst = x_word_feature.shape[0]

        lil = x_word_feature.tolil()

        indptr = [0]
        indices = []
        data = []
        for i, line in enumerate(X):
            for col, val in zip(lil.rows[i], lil.data[i]):
                indices.append(col)
                data.append(val)
            if self.use_char_ngram:
                new_line = Counter()
                for t in find_ngram_range(line, range(2,6)):
                    if t in self.char_feature_dict:
                        f_id = self.c_base + self.char_feature_dict[t]
                        new_line[f_id] += 1

                for key, val in new_line.items():
                    indices.append(key)
                    data.append(val)
            indptr.append(len(indices))

        X_out = csr_matrix((data, indices, indptr), dtype=int,
                           shape=(num_inst, self.num_feature))
        return X_out

    def fit_transform(self, X):
        self.char_feature_dict = dict()
        feature_id = 0
        if self.use_char_ngram:
            for line in X:
                tokens = line.lower()
                for t in find_ngram_range(line, range(2,6)):
                    if t not in self.char_feature_dict:
                        self.char_feature_dict[t] = feature_id
                        feature_id += 1

        self.word_feature = CountVectorizer(ngram_range=(1, self.ngram_max))
        self.word_feature.fit(X)

        n_gram_num = len(self.word_feature.vocabulary_)
        print("word feature : ", n_gram_num)
        self.c_base = n_gram_num + 1
        self.num_feature = self.c_base + len(self.char_feature_dict)
        print("total feature : ", self.num_feature)
        return self.transform(X)

