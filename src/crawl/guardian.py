import os
import requests
import json
import time
from path import data_path
import pickle

def ask_list_with_body(query, page):
    page_str = str(page)
    url = "http://content.guardianapis.com/search?q={}&show-fields=all&page-size=200&page={}"\
        .format(query,page_str)
    apikey = "c13d9515-b19e-412b-b505-994677cc2cf3"

    headers = {
        "api-key": apikey,
        "format": "json",
    }
    res = requests.get(url, headers)
    if res.status_code == 200:
        return res.content
    else :
        print(res.content)
        return None

save_dir = os.path.join(data_path, "guardian", "topic")

def save_query_result(topic, page, content):
    topic_dir = os.path.join(save_dir, topic)
    if not os.path.exists(topic_dir):
        os.mkdir(topic_dir)
    file_name = "{}.json".format(page)

    path = os.path.join(topic_dir, file_name)
    open(path, "wb").write(content)

def get_topic_list(small=False):
    path = os.path.join(data_path, "controversy", "clueweb", "tf10_all.pickle")
    if small:
        path = os.path.join(data_path, "controversy", "clueweb", "tf10.pickle")
    obj = pickle.load(open(path, "rb"))
    all_terms = set()
    for doc, terms in obj:
        all_terms.update(terms)
    return all_terms

def crawl_by_list():
    topic_list = get_topic_list()
    for topic in topic_list:
        print(topic)
        content = ask_list_with_body(topic, 1)
        j = json.loads(content)
        num_pages = j['response']['pages']
        save_query_result(topic, 1, content)

        #for page_no in range(2,num_pages+1):
        #    ask_list_with_body(topic, page_no)
        #    save_query_result(topic, page_no, content)

        time.sleep(0.1)

def get_comment(short_id):
    url_prefix = "http://discussion.guardianapis.com/discussion-api/discussion/"

    url = url_prefix + short_id
    print(url, end=" ")
    res = requests.get(url)
    if res.status_code == 200 :
        print("success ")
        return res.content
    elif res.status_code == 404:
        if json.loads(res.content)["errorCode"] == "DISCUSSION_NOT_FOUND":
            print("DISCUSSION_NOT_FOUND")
            return None
        else:
            print(res.content)
    else:
        print(res.content)
        return None

scope_dir = os.path.join(data_path, "guardian")

def save_comment(short_id, content):
    file_name = short_id.replace("/", "_") + ".json"
    f = open(os.path.join(scope_dir, "comments", file_name), "wb")
    f.write(content)
    f.close()

def load_short_ids(topic):
    topic_dir = os.path.join(save_dir, topic)
    file_name = "{}.json".format(1)
    path = os.path.join(topic_dir, file_name)
    j = json.load(open(path, "rb"))
    r = j['response']['results']
    id_list = []
    for item in r:
        id = item['id']
        shortUrl = item['fields']['shortUrl']
        id_list.append((id, shortUrl))

    short_ids = []
    for id, shortUrl in id_list:
        short_ids.append(shortUrl[14:].strip())

    return short_ids


def crawl_comments():
    topic_list = get_topic_list()
    logging_path = os.path.join(scope_dir, "comment_log.pickle")

    acquire_list = pickle.load(open(logging_path, "rb"))
    print("Already crawled : ", len(acquire_list))
    def update_acquired():
        pickle.dump(acquire_list, open(logging_path, "wb"))

    for topic in topic_list:
        if topic in acquire_list:
            continue
        print(topic)
        try:
            aid_list = load_short_ids(topic)
            for short_id in aid_list[:100]:
                acquire_list.add(short_id)
                r = get_comment(short_id)
                if r is not None:
                    save_comment(short_id, r)
        except FileNotFoundError as e:
            print("skip ", topic)
        update_acquired()


if __name__ == "__main__":
    #crawl_by_list()
    crawl_comments()