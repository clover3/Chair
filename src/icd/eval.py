import gensim
import nltk
from gensim.models import KeyedVectors

from icd.common import load_description, lmap, AP_from_binary


def do_eval(w2v):
    data = load_description()

    ids = set(lmap(lambda x: x['icd10_code'].strip(), data))
    ap_list= []
    for e in data[:1000]:
        word = e['icd10_code'].strip()
        terms = nltk.word_tokenize(e['short_desc'])
        ranked_list = list([w for w in w2v.most_similar(word, topn=50) if w[0] not in ids])

        def is_correct(w_pair):
            return w_pair[0] in terms
        AP = AP_from_binary(lmap(is_correct, ranked_list), len(terms))
        ap_list.append(AP)

    print("MAP", sum(ap_list)/ len(ap_list))


def eval_word2vec():
    save_name = "sent2.w2v"
    w2v = gensim.models.Word2Vec.load(save_name)
    do_eval(w2v)


def eval_manual_vec():
    mv = KeyedVectors.load_word2vec_format("manual_voca.txt", binary=False)
    do_eval(mv)


if __name__ == "__main__":
    eval_word2vec()