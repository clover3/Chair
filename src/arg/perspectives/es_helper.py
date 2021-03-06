from elasticsearch import Elasticsearch

from elastic.elastic_info import auth_info

server = "gosford.cs.umass.edu:9200"

es = Elasticsearch(server,
                   http_auth=auth_info
                   )


def get_perspective_from_pool(text, size):
    res = es.search(index="perspective_pool",
                    body={"query": {"match": {"text": text}}},
                    doc_type="text",
                    size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        pId = doc['_source']["pId"]
        output.append((text, pId, score))
    return output


def get_perspective_from_pool2(text, size):
    res = es.search(index="perspective_pool", doc_type="text", body={"query": {"match": {"text": text}}}, size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        pId = doc['_source']["pId"]
        output.append((text, pId, score))
    return output
