from nltk import ngrams
from scipy.special import softmax

from arg.claim_building.count_ngram import merge_subword
from arg.claim_building.count_stance_prediction import enum_docs_and_stance
from arg.stance_build import get_all_term_odd
from cache import load_from_pickle
from misc_lib import left
from models.classic.stopword import load_stopwords


def get_odd_list():
    result = load_from_pickle("majority_3gram")
    tf0, tf1, tf2 = result
    odd_dict = get_all_term_odd(tf1, tf2, 0.95)

    def contrib(e):
        key, value = e
        return (tf1[key] + tf2[key]) * value

    odd_list = list(odd_dict.items())
    odd_list.sort(key=contrib, reverse=True)
    stopword = load_stopwords()

    def valid(e):
        key, value = e
        return key not in stopword and tf1[key] > 10 and tf2[key] > 10

    acc = 0
    for key, value in odd_list:
        acc += value * (tf1[key] + tf2[key])

    ctf = sum(tf1.values()) + sum(tf2.values())
    print(acc, acc/ctf)

    return list(filter(valid, odd_list))


def get_top_cont_ngram():
    odd_list = get_odd_list()
    k = 50
    return left(odd_list[:k])


def display():
    target_n_gram = get_top_cont_ngram()
    for doc, preds in enum_docs_and_stance():
        assert len(preds) == len(doc)
        for sent, pred in zip(doc, preds):
            probs = softmax(pred)

            sent = merge_subword(sent)
            do_print = False
            found_ngram = []
            for ngram in ngrams(sent, 3):
                if ngram in target_n_gram :
                    do_print = True
                    found_ngram.append(ngram)

            if do_print:
                print(probs)
                print(found_ngram)
                print(" ".join(sent))


if __name__ == "__main__":
    display()