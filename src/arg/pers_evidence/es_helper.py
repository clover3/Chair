from elasticsearch import Elasticsearch

from elastic.elastic_info import auth_info

server = "gosford.cs.umass.edu:9200"
es = Elasticsearch(server, http_auth=auth_info)
index_name = "perspectrum_evidence"


def get_evidence_from_pool(text, size):
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
