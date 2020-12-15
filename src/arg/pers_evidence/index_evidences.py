from elasticsearch import Elasticsearch

from arg.perspectives.load import load_evidence_dict
from elastic.elastic_info import auth_info

server = "gosford.cs.umass.edu:9200"

es = Elasticsearch(server, http_auth=auth_info)

index_name = "perspectrum_evidence"


def insert():
    e_dict = load_evidence_dict()
    for e_id, text in e_dict.items():
        p = {
            'id': e_id,
            'text': text
        }
        r = es.index(index=index_name,  body=p)


def search_test():
    text = "vaccination should be compulsory"
    p = {"query": {"match": {"text": text}}}
    r = es.search(index=index_name, body=p)
    print(r)


if __name__ == "__main__":
    search_test()
