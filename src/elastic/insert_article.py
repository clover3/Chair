from elasticsearch import Elasticsearch
from crawl.guardian_uk import load_commented_articles_opinion, load_commented_articles_any, get_list_url_claim3


def hello():
    es = Elasticsearch("localhost")
    r = es.get(index="guardian", id="1")
    print(r)


def insert_article():
    articles = load_commented_articles_opinion()
    print("{} articles".format(len(articles)))
    server_name = "gosford.cs.umass.edu"
    es = Elasticsearch(server_name)

    for article in articles:
        id, title, short_id, infos = article
        r = es.index(index="guardian", body=infos)



def insert_white_list_articles():
    urls = get_list_url_claim3()
    print(urls)
    articles = load_commented_articles_any()

    selected = []
    for a in articles:
        id, title, short_id, infos = a
        if infos['webUrl'] in urls:
            selected.append(a)

    print("TOtal of {} articles " .format(len(selected)))
    server_name = "gosford.cs.umass.edu"
    es = Elasticsearch(server_name)
    for article in articles:
        id, title, short_id, infos = article
        r = es.index(index="guardian", body=infos)


insert_white_list_articles()

