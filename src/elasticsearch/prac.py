from elasticsearch import Elasticsearch
from crawl.guardian_uk import load_commented_articles


def hello():
    es = Elasticsearch("128.119.40.196")
    r = es.get(index="guardian", id="1")
    print(r)


def insert_test():
    articles = load_commented_articles()
    es = Elasticsearch("10.3.40.184")

    for article in articles:
        id, title, short_id, infos = article
        r = es.index(index="guardian",body=infos)

hello()
#insert_test()

