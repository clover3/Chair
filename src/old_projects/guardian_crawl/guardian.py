import calendar
import json
import os
import pickle
import time
from datetime import datetime

import requests

from cpath import data_path
from old_projects.guardian_crawl.guardian_api import get_comment, load_short_ids_from_path


def ask_list_with_body(query, page):
    page_str = str(page)
    url = "http://content.guardianapis.com/search?q={}&show-fields=all&page-size=200&page={}"\
        .format(query,page_str)

    #url = "http://content.guardianapis.com/search?to-date=2009-12-30&q={}&show-fields=all&page-size=200&page={}" \
    #    .format(query, page_str)
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

save_dir = os.path.join(data_path, "guardian", "controversy")

def save_query_result(topic, page, content):
    #topic = topic.replace(" ", "_")
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
    #topic_list = get_topic_list()
    topic_list = ["2020 census citizenship"]
    for topic in topic_list:
        print(topic)
        content = ask_list_with_body(topic, 1)
        j = json.loads(content)
        num_pages = j['response']['pages']
        save_query_result(topic, 1, content)

        for page_no in range(2,num_pages+1):
            ask_list_with_body(topic, page_no)
            save_query_result(topic, page_no, content)

        time.sleep(0.1)


scope_dir = os.path.join(data_path, "guardian")

def save_comment(short_id, content):
    file_name = short_id.replace("/", "_") + ".json"
    f = open(os.path.join(scope_dir, "comments_2009", file_name), "wb")
    f.write(content)
    f.close()


def load_short_ids(topic):
    topic_dir = os.path.join(save_dir, topic)
    file_name = "{}.json".format(1)
    path = os.path.join(topic_dir, file_name)
    return load_short_ids_from_path(path)

def crawl_comments(topic_list, logging_path):


    #acquire_list = pickle.load(open(logging_path, "rb"))
    acquire_list = set()
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


def get_opinion_article(query, page_no):
    page_str = str(page_no)
    url = "https://content.guardianapis.com/search?section=commentisfree" \
          "&from-date=2019-01-01&to-date=2019-06-30" \
          "&page-size=200" \
          "&page={}" \
          "&show-fields=bodyText%2CshortUrl%2Cbody" \
          "&q={}"\
        .format(page_str, query)
    print(url)
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

def get_any_article(query, page_no):
    page_str = str(page_no)
    url = "https://content.guardianapis.com/search?" \
          "&from-date=2019-01-01&to-date=2019-06-30" \
          "&page-size=200" \
          "&page={}" \
          "&show-fields=bodyText%2CshortUrl%2Cbody" \
          "&q={}"\
        .format(page_str, query)
    print(url)
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


def get_article_list(time_from, time_to, page_no):
    page_str = str(page_no)
    url = "https://content.guardianapis.com/search?" \
          "&from-date={}&to-date={}" \
          "&page-size=200" \
          "&show-fields=shortUrl" \
          "&page={}" \
        .format(time_from, time_to, page_str)
    print(url)
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



def crawl_opinion_articles(topic):
    save_dir = os.path.join(data_path, "guardian", "opinion")

    def save_query_result(topic, page, content):
        # topic = topic.replace(" ", "_")
        topic_dir = os.path.join(save_dir, topic)
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        file_name = "{}.json".format(page)

        path = os.path.join(topic_dir, file_name)
        open(path, "wb").write(content)

    content = get_opinion_article(topic, 1)
    j = json.loads(content)
    num_pages = j['response']['pages']
    save_query_result(topic, 1, content)

    for page_no in range(2, num_pages + 1):
        content = get_opinion_article(topic, page_no)
        save_query_result(topic, page_no, content)

    time.sleep(0.1)




def crawl_articles(topic):
    save_dir = os.path.join(data_path, "guardian", "any")

    def save_query_result(topic, page, content):
        # topic = topic.replace(" ", "_")
        topic_dir = os.path.join(save_dir, topic)
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        file_name = "{}.json".format(page)

        path = os.path.join(topic_dir, file_name)
        open(path, "wb").write(content)

    content = get_any_article(topic, 1)
    j = json.loads(content)
    num_pages = j['response']['pages']
    save_query_result(topic, 1, content)

    for page_no in range(2, num_pages + 1):
        content = get_any_article(topic, page_no)
        save_query_result(topic, page_no, content)

    time.sleep(0.1)


def save_by_time(year, month, page, content):
    dir_name = "{}-{}".format(year, month)
    dir_path = os.path.join(scope_dir, "by_time", dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_name = "{}.json".format(page)
    path = os.path.join(dir_path, file_name)
    open(path, "wb").write(content)


def crawl_by_time(year, month):
    begin = datetime(year, month, 1)
    d1, dl = calendar.monthrange(year, month)
    end = datetime(year, month, dl)
    time_from = begin.strftime("%Y-%m-%d")
    time_to = end.strftime("%Y-%m-%d")
    content = get_article_list(time_from, time_to, 1)
    j = json.loads(content)
    num_pages = j['response']['pages']
    save_by_time(year, month, 1, content)
    for page_no in range(2, num_pages + 1):
        content = get_article_list(time_from, time_to, page_no)
        save_by_time(year, month, page_no, content)




def craw_all_articles():
    for year in range(2019, 2010, -1):
        for month in range(1,13):
            print(year, month)
            crawl_by_time(year, month)



if __name__ == "__main__":
    craw_all_articles()