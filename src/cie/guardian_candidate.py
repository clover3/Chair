from random import shuffle

import nltk

from cie.candidate import get_verb_nouns, get_entities
from crawl.rerank_guardian import load_all_articles, load_ranking
from list_lib import left
from misc_lib import *


def get_top_ids(score_list, top_k):
    unique_ids = set()
    for id, title, score in score_list:
        if id not in unique_ids:
            print(id, score)
        unique_ids.add(id)
        if score < 2:
            break
        if len(unique_ids) > top_k:
            break
    return unique_ids

def guardian_generate(query):
    articles = load_all_articles(query)
    article_d = {}
    for entry in articles:
        id = entry[0]
        article_d[id] = entry

    score_list = load_ranking(query)

    top_k = 200
    ids = get_top_ids(score_list, top_k)
    sents = []
    for id in ids:
        print(id)
        id, title, short_id, text = article_d[id]
        sents += [title] + nltk.sent_tokenize(text)

    verbs_all = Counter()
    nouns_all = Counter()
    entities_all = Counter()
    print("POS tagging...")
    shuffle(sents)
    size_small = int(len(sents)*0.1)
    ticker = TimeEstimator(size_small)
    sub_sents = sents[:size_small]
    for sent in sub_sents:
        verbs, nouns = get_verb_nouns(sent)
        nouns_all.update(nouns)
        verbs_all.update(verbs)
        entities_all.update(get_entities(sent))
        ticker.tick()


    v_top = left(verbs_all.most_common(100))
    n_top = left(nouns_all.most_common(100))
    en_top = left(entities_all.most_common(100))

    print("Verbs")
    list_print(v_top, 10)
    print("Nouns")
    list_print(n_top, 10)
    print("Entities")
    list_print(en_top, 10)

def interactive(query):
    articles = load_all_articles(query)
    article_d = {}
    for entry in articles:
        id = entry[0]
        article_d[id] = entry

    score_list = load_ranking(query)
    sents = []

    unique_ids = get_top_ids(score_list, 200)

    for id in unique_ids:
        id, title, short_id, text = article_d[id]
        sents += [title] + nltk.sent_tokenize(text)

    while query != "finish":
        shuffle(sents)
        matched = []
        for sent in sents:
            match = True
            for q in query.split():
                if q not in sent:
                    match = False
                    break
            if match:
                matched.append(sent)
            if len(matched) > 30:
                break

        print("Query : " + query)
        for sent in matched:
            print(sent)

        print("Enter next query : ")
        query = input()

if __name__ == "__main__":
    interactive("2020 census citizenship")
