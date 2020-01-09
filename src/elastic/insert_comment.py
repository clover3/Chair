import os

from elasticsearch import Elasticsearch

from cpath import data_path
from crawl import parse_comment
from misc_lib import get_dir_files


def load_all_comments(dir_path):
    for comment_path in get_dir_files(dir_path):
        yield parse_comment.parse_comments(comment_path)



def load_guardian_uk_comments():
    save_dir = os.path.join(data_path, "guardian", "opinion")
    topic = "UK"
    topic_save_dir = os.path.join(save_dir, topic)
    comments_dir = os.path.join(save_dir, "comments")
    return load_all_comments(comments_dir)

def insert_uk_comments():
    server_name = "gosford.cs.umass.edu"
    es = Elasticsearch(server_name)
    data = load_guardian_uk_comments()

    for comment in data:
        print(comment)
        r = es.index(index="guardian_comment",body=comment)


def insert_comment_piece():
    es = Elasticsearch("localhost")
    data = load_guardian_uk_comments()

    for comment in data:
        r = comment['comments']
        short_id = comment['short_id']
        for e in r:
            head, tail = e
            p = {'id':head[0], 'text':head[1], 'dicsussion_id':short_id}
            r = es.index(index="guardian_comment_piece", body = p)
            for t in tail:
                p = {'id':t[0], 'text':t[1], 'discussion_id':short_id}
                r = es.index(index="guardian_comment_piece", body=p)

        print()

