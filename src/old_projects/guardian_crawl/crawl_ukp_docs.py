import os
import pickle

import requests

import cpath
from data_generator.argmining.document_stat import load_with_doc


def get_list_of_urls_per_topic():
    all_data = load_with_doc()
    for topic in all_data:
        topic_data = all_data[topic]
        yield topic, list(topic_data.keys())


def save_urls(url_list, save_path):
    doc_dict = {}
    for url in url_list:
        ret = requests.get(url)
        if ret.status_code != 200:
            print(url, ret.status_code)
        else:
            doc_dict[url] = ret.content

    pickle.dump(doc_dict, open(save_path, "wb"))

def get_topic_doc_save_path(topic):
    return os.path.join(cpath.data_path, "ukp_docs", topic)

def main():
    for topic, urls in get_list_of_urls_per_topic():
        save_path = get_topic_doc_save_path(topic)
        print(topic)
        save_urls(urls, save_path)



if __name__ == "__main__":
    main()

