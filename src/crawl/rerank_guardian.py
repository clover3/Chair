import os
from path import data_path
from misc_lib import *
from dispute.guardian import load_article_w_title
from adhoc.bm25 import stem_tokenize, BM25_2
from adhoc.idf import Idf_tokens
from collections import Counter
import pickle

def get_file_list(query):
    save_dir = os.path.join(data_path, "guardian", "controversy")
    topic_dir = os.path.join(save_dir, query)
    return get_dir_files(topic_dir)


def load_all_articles(query):
    files = get_file_list(query)
    all_articles = []
    for json_path in files:
        articles = load_article_w_title(json_path)
        all_articles.extend(articles)

    print("Total of {} articles".format(len(all_articles)))
    return all_articles


def load_ranking(query):
    save_dir = os.path.join(data_path, "guardian", "controversy")

    name = query + ".rank.pickle"
    save_path = os.path.join(save_dir, name)
    score_pair = pickle.load(open(save_path, "rb"))
    return score_pair



def rerank(query):
    articles = load_all_articles(query)
    articles = articles
    token_docs = {}
    ticker = TimeEstimator(len(articles))


    for id, title, short_id, text in articles:
        # repeat title twice
        doc_rep = title + " " + title + " " + text
        token_docs[id] = stem_tokenize(doc_rep)
        ticker.tick()

    avdl = average(list([len(t) for t in token_docs.values()]))
    idf = Idf_tokens(token_docs.values())
    N = len(articles)

    q_terms = stem_tokenize(query)

    score_pair = []
    for id, title, short_id, text in articles:
        tf_d = Counter(token_docs[id])

        score = 0
        for q_term in q_terms:
            score += BM25_2(tf_d[q_term], idf.df[q_term], N, N, avdl)
        score_pair.append((id, title, score))

    score_pair.sort(key=lambda x:x[2], reverse=True)

    save_dir = os.path.join(data_path, "guardian", "controversy")

    name = query + ".rank.pickle"
    save_path = os.path.join(save_dir, name)
    pickle.dump(score_pair, open(save_path, "wb"))



if __name__ == "__main__":
    rerank("2020 census citizenship")
