from path import data_path
import os
from crawl.load_guardian import load_articles_from_dir

def load_commented_articles():
    save_dir = os.path.join(data_path, "guardian", "opinion")
    topic = "UK"
    topic_save_dir = os.path.join(save_dir, topic)
    comments_dir = os.path.join(save_dir, "comments")

    articles = load_articles_from_dir(topic_save_dir)
    commented_articles = []
    for article in articles:
        id, title, short_id, infos = article
        comment_path = os.path.join(comments_dir, short_id.replace("/" ,"_"))
        if os.path.exists(comment_path):
            commented_articles.append(article)
    return commented_articles



def save():

    a_list = load_commented_articles()
    import json

    p  = os.path.join(data_path, "guardian.json")
    json.dump(a_list, open(p, "w"))


save()