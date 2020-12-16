from elastic.elastic_info import auth_info

es = None
index_name = "perspectrum_evidence"


def init_es():
    server = "gosford.cs.umass.edu:9200"
    from elasticsearch import Elasticsearch
    global es
    es = Elasticsearch(server, http_auth=auth_info)


def get_evidence_from_pool(text, size):
    if es is None:
        init_es()
    res = es.search(index=index_name,
                    body={"query": {"match": {"text": text}}},
                    size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        eId = doc['_source']["id"]
        output.append((text, eId, score))
    return output
