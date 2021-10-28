import nltk

from cache import load_from_pickle
from dataset_specific.msmarco.common import load_queries
from models.classic.stopword import load_stopwords_for_query


class FunctionWordClassifier:
    def __init__(self):
        self.qdf = load_from_pickle("msmarco_qdf")
        self.stopwords = load_stopwords_for_query()


    def score(self, word):
        qdf_score = self.qdf[word]
        is_stopword_score = 1000 if word in self.stopwords else 0
        return qdf_score + is_stopword_score

    def is_function_word(self, word):
        return self.score(word) >= 1000


def print_top_words():
    cls = FunctionWordClassifier()
    entries = []
    for term in cls.qdf:
        qdf = cls.qdf[term]
        score = cls.score(term)
        e = (term, score, qdf)
        entries.append(e)
    entries.sort(key=lambda x: x[1], reverse=True)

    for term, score, qdf in entries[:1000]:
        print(term, score, qdf)


def dev_demo():
    cls = FunctionWordClassifier()
    queries = load_queries("dev")
    for qid, q_str in queries:
        q_tokens = nltk.word_tokenize(q_str)
        function_words = list(filter(cls.is_function_word, q_tokens))
        content_words = [t for t in q_tokens if t not in function_words]
        print("{} / {}".format(" ".join(function_words), " ".join(content_words)))



if __name__ == "__main__":
    dev_demo()
