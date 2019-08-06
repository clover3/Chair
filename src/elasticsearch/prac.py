from elasticsearch import Elasticsearch
from crawl.guardian_uk import load_commented_articles


def hello():
    es = Elasticsearch("localhost")
    r = es.get(index="guardian", id="1")
    print(r)


def insert_test():
    articles = load_commented_articles()
    es = Elasticsearch("localhost")

    for article in articles:
        id, title, short_id, infos = article
        r = es.index(index="guardian",body=infos)

#hello()
insert_test()

