from elasticsearch import Elasticsearch

from arg.counter_arg.enum_all_argument import enum_all_argument
from arg.counter_arg.header import splits
from elastic.elastic_info import auth_info

server = "gosford.cs.umass.edu:9200"

es = Elasticsearch(server,
                   http_auth=auth_info
                   )


def get_index_name(split):
    return "argu_{}".format(split)


def insert(split):
    index_name = get_index_name(split)
    for argu in enum_all_argument(split):
        p = {
            'id': argu.id.id,
            'text': argu.text
        }
        r = es.index(index=index_name,  body=p)


def search(split, query, size):
    res = es.search(index=get_index_name(split),
                    body={"query": {"match": {"text": query}}}, size=size)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        data_id = doc['_source']["id"]
        output.append((text, data_id, score))
    return output


def search_by_id(split, query):
    res = es.search(index=get_index_name(split),
                    body={"query": {"match": {"id": query}}}, size=1)
    output = []
    for doc in res['hits']['hits']:
        score = doc['_score']
        text = doc['_source']["text"]
        data_id = doc['_source']["id"]
        output.append((text, data_id, score))
    return output


if __name__ == "__main__":
    #r = search("training", "hello world", 10)
    #insert("training")
    for split in splits[1:]:
        insert(split)
