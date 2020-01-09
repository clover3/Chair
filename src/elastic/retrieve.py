from elasticsearch import Elasticsearch

server = "gosford.cs.umass.edu"

def get_comment(doc_id, comment_id):
    es = Elasticsearch(server)
    q = {
        "query": {
            "term": {"short_id.keyword": doc_id}
        }
    }
    comment_id = str(comment_id)
    r =es.search(index="guardian_comment", body=q)
    discussion = r['hits']['hits'][0]

    for thread in discussion['_source']['comments']:
        head, tail = thread

        h_c_id, h_text = head
        if h_c_id == comment_id:
            return h_text
        for t in tail:
            t_c_id, c_text, c_target = t
            if t_c_id == comment_id:
                return c_text
    print(discussion)
    raise Exception("Not found : ", doc_id, comment_id)


def get_paragraph(doc_id, para_id):
    es = Elasticsearch(server)
    q = {
        "query": {
            "term": {"short_id.keyword": doc_id}
        }
    }

    r = es.search(index="guardian", body=q)
    r = r['hits']['hits'][0]
    data = r['_source']
    assert doc_id == data['short_id']

    return data['paragraphs'][para_id]



def get_title(doc_id):
    es = Elasticsearch(server)
    q = {
        "query": {
            "term": {"short_id.keyword": doc_id}
        }
    }

    r =es.search(index="guardian", body=q)
    r = r['hits']['hits'][0]
    data = r['_source']
    assert doc_id == data['short_id']
    return data['webTitle']

