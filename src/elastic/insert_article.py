from elasticsearch import Elasticsearch
from crawl.guardian_uk import load_commented_articles_opinion


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


#hello()
insert_article()

