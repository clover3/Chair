import json
import os


def load_article_w_title(path):
    f = open(path, encoding='utf-8')
    j = json.load(f)
    articles = []
    for j_article in j['response']['results']:
        id = j_article['id']
        title = j_article['webTitle']
        body_text = j_article['fields']['bodyText']
        short_url = j_article['fields']['shortUrl']

        infos = {}
        for key in j_article:
            if key != "fields":
                infos[key] = j_article[key]
        infos["bodyText"] = body_text
        short_id = short_url[-len("/p/ap83f"):]
        articles.append((id, title, short_id, infos))
    return articles


def load_articles_from_dir(dir_path):
    idx = 1
    def get_next_path():
        file_name = "{}.json".format(idx)
        return os.path.join(dir_path, file_name)

    all_articles = []
    while os.path.exists(get_next_path()):
        target_path = get_next_path()
        all_articles.extend(load_article_w_title(target_path))
        idx += 1

    return all_articles