from path import data_path
import os
from crawl.load_guardian import load_articles_from_dir
from collections import Counter

def load_commented_articles_opinion():
    save_dir = os.path.join(data_path, "guardian", "opinion")
    return load_commented_articles(save_dir)


def load_commented_articles(save_dir):
    topic = "UK"
    topic_save_dir = os.path.join(save_dir, topic)
    comments_dir = os.path.join(save_dir, "comments")

    print("Loading Articles...")
    articles = load_articles_from_dir(topic_save_dir)
    print("comments ")
    commented_articles = []
    for article in articles:
        id, title, short_id, infos = article
        comment_path = os.path.join(comments_dir, short_id.replace("/" ,"_"))
        if os.path.exists(comment_path):
            commented_articles.append(article)
    return commented_articles



def save():
    a_list = load_commented_articles_opinion()
    import json

    p = os.path.join(data_path, "guardian.json")
    json.dump(a_list, open(p, "w"))


def save_url_list():
    dir_path = os.path.join(data_path, "guardian", "any")
    a_list = load_commented_articles(dir_path)
    save_path = os.path.join(dir_path, "urls.txt")
    f = open(save_path, 'w')

    c = Counter()
    for a in a_list:
        id, title, short_id, infos = a
        #print(title.strip(), infos['webUrl'])
        print(infos['sectionName'])
        c[infos['sectionName']] += 1
        if infos['sectionName'] != "Opinion":
            f.write("{}\t{}\n".format(title.strip(), infos['webUrl']))
    print(c)

if __name__ == "__main__":
    save_url_list()

