from path import data_path
import os
from misc_lib import get_dir_files
from crawl import parse_comment
from elasticsearch import Elasticsearch



def get_comment(doc_id, comment_id):
    es = Elasticsearch("gosford.cs.umass.edu")
    q = {
        "query": {
            "term": {"short_id.keyword": doc_id}
        }
    }

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


def get_paragraph(doc_id, para_id):
    es = Elasticsearch("gosford.cs.umass.edu")
    q = {
        "query": {
            "term": {"short_id.keyword": doc_id}
        }
    }

    r =es.search(index="guardian", body=q)
    r = r['hits']['hits'][0]
    data = r['_source']
    assert doc_id == data['short_id']

    return data['paragraphs'][para_id]

