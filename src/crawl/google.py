import json
import pickle

import requests
from bs4 import BeautifulSoup

from cpath import data_path
from misc_lib import *

scope_path = os.path.join(data_path, "arg", "extend")
serp_path = os.path.join(scope_path, "serp")
from collections import defaultdict

def parse_html(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    web_result = soup.find('div', attrs = {'class':'srg'})
    inst_list = web_result.find_all('div', attrs = {'class':'g'})
    for inst in inst_list:
        r = inst.find('div', attrs = {'class':'r'}).find('a')
        yield r['href']


def parse_urls(doc_name):
    full_path = os.path.join(serp_path, doc_name)
    return list(parse_html(open(full_path).read()))


def parse_all_urls():
    docs = []
    for (dirpath, dirnames, filenames) in os.walk(serp_path):
        docs.extend(filenames)

    for name in docs:
        yield parse_urls(name)

def get_topic_urls_dicts():
    r = defaultdict(list)

    docs = []
    for (dirpath, dirnames, filenames) in os.walk(serp_path):
        docs.extend(filenames)

    for page_name in docs:
        topic = page_name.split(" - ")[0].strip()
        urls = parse_urls(page_name)
        r[topic].extend(urls)
    return r


way_back_save_path = os.path.join(data_path, "arg", "extend", "wayback", "res.pickle")
def save_way_back_fetch():
    all_url = flatten(parse_all_urls())
    wayback_dict = {}
    for url in all_url:
        print(url)
        prefix = "http://archive.org/wayback/available?url="
        ret = requests.get(prefix + url)
        if ret.status_code != 200:
            print(ret.status_code)
            break
        else:
            wayback_dict[url] = ret.content

    pickle.dump(wayback_dict, open(way_back_save_path, "wb"))


def parse_wayback_response(html):
    soup = BeautifulSoup(html, 'html.parser')
    print(soup)
    print(soup.find('calendar-layout'))
    valid_days = soup.find_all('div', attrs = {'class':'calendar-day'})
    for day in valid_days:
        print(day)


def parse_way_back():
    wayback_dict = pickle.load(open(way_back_save_path, "rb"))
    valid_match = dict()
    all_urls = list(wayback_dict.keys())
    cnt = 0
    for key in all_urls:
        j = json.loads(wayback_dict[key])
        if j['archived_snapshots']:
            valid_match[key] = j['archived_snapshots']['closest']['url']
            cnt += 1
    print("{} of {} has snapshot".format(cnt, len(all_urls)))
    return valid_match

def save_as_tsv():
    topic_urls = get_topic_urls_dicts()
    url_to_archieve = parse_way_back()

    out_dir = os.path.join(scope_path, "urls")
    for topic in topic_urls.keys():
        filename = topic.lower().replace(" ", "_") + ".tsv"
        f = open(os.path.join(out_dir, filename), "w")
        cnt = 0
        for url in topic_urls[topic]:
            if url in url_to_archieve:
                f.write("{}\t{}\n".format(url, url_to_archieve[url]))
                cnt += 1

        print(topic, cnt)



if __name__ == "__main__":
    save_as_tsv()