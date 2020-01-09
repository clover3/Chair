import os
import pickle

from cpath import data_path
from crawl.guardian_api import *

save_dir = os.path.join(data_path, "guardian", "any")



def load_short_ids_from_article_dir(dir_path):
    idx = 1
    def get_next_path():
        file_name = "{}.json".format(idx)
        return os.path.join(dir_path, file_name)

    all_short_ids = []
    while os.path.exists(get_next_path()):
        target_path = get_next_path()
        short_ids = load_short_ids_from_path(target_path)
        all_short_ids.extend(short_ids)
        idx += 1
    return all_short_ids


def crawl_comments(topic_list, comments_dir, logging_path):
    if os.path.exists(logging_path):
        acquire_list = pickle.load(open(logging_path, "rb"))
        print("Already crawled : ", len(acquire_list))
    else:
        acquire_list = set()
    def update_acquired():
        pickle.dump(acquire_list, open(logging_path, "wb"))

    def save_comment(short_id, r):
        save_path = os.path.join(comments_dir, short_id.replace("/", "_"))
        pickle.dump(r, open(save_path, "wb"))

    for topic in topic_list:
        if topic in acquire_list:
            continue
        print(topic)
        try:
            topic_save_dir = os.path.join(save_dir, topic)
            aid_list = load_short_ids_from_article_dir(topic_save_dir)
            for short_id in aid_list:
                acquire_list.add(short_id)
                r = get_comment(short_id)
                if r is not None:
                    save_comment(short_id, r)
        except FileNotFoundError as e:
            print("skip ", topic)
        update_acquired()

def crawl_uk_comment():
    logging_path = os.path.join(save_dir, "comment_log_uk.pickle")
    comments_dir = os.path.join(save_dir, "comments")
    crawl_comments(["UK"], comments_dir, logging_path)





if __name__ == "__main__":
    crawl_uk_comment()