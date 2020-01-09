import os

from cpath import data_path
from crawl.load_guardian import load_articles_from_dir

save_dir = os.path.join(data_path, "guardian", "opinion")

def view_commented():
    topic = "UK"
    topic_save_dir = os.path.join(save_dir, topic)
    comments_dir = os.path.join(save_dir, "comments")

    articles = load_articles_from_dir(topic_save_dir)
    print("{} articles ".format(len(articles)))
    for article in articles:
        id, title, short_id, infos = article
        comment_path = os.path.join(comments_dir, short_id.replace("/" ,"_"))
        if os.path.exists(comment_path):
            print(title)


view_commented()
