from path import data_path
import os
from misc_lib import get_dir_files
from crawl import parse_comment
from elasticsearch import Elasticsearch

def load_all_comments(dir_path):
    for comment_path in get_dir_files(dir_path):
        print(comment_path)
        yield parse_comment.parse_comments(comment_path)



def load_guardian_uk_comments():
    save_dir = os.path.join(data_path, "guardian", "opinion")
    topic = "UK"
    topic_save_dir = os.path.join(save_dir, topic)
    comments_dir = os.path.join(save_dir, "comments")
    return load_all_comments(comments_dir)

def insert_uk_comments():
    es = Elasticsearch("localhost")
    data = load_guardian_uk_comments()

    for comment in data:
        print(comment)
        r = es.index(index="guardian_comment",body=comment)



insert_uk_comments()