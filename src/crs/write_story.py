import json
import os

from cpath import data_path
from crawl.guardian_uk import load_commented_articles_opinion
from crs.load_claim_annotation import load_claim_annot
from elastic.insert_comment import load_guardian_uk_comments


def write_guardian_claims(name, add_comment=False):
    file_path = os.path.join(data_path, "guardian", "claim", name)
    out_dir = os.path.join(data_path, "guardian", "claim_story_c")
    data = load_claim_annot(file_path)
    articles = load_commented_articles_opinion()

    if add_comment:
        comments = load_guardian_uk_comments()
        c_dict = {}
        for c in comments:
            threads = c['comments']
            short_id = c['short_id']
            for thread in threads:
                head, tail = thread
                texts = [head[1]]
                for t in tail:
                    texts.append(t[1])

                if len(texts) > 200:
                    break
            c_dict[short_id] = texts

    def write(short_id, title, paras, statements):
        out_path = os.path.join(out_dir, short_id.replace("/", "_") + ".story")
        f = open(out_path, "w")
        f.write(title + '\n\n')
        for p in paras:
            f.write(p + "\n\n")

        for s in statements:
            f.write("@highlight\n\n")
            f.write(s +'\n\n')
        f.close()

    long_url_indice = {}
    for a in articles:
        id, title, short_id, infos = a
        w = infos['webUrl']
        long_url_indice[w] = a

    for entry in data:
        webUrl = entry['url']
        article = long_url_indice[webUrl]
        id, title, short_id, infos = article
        paras = infos["paragraphs"]
        statements = entry['statements']

        contents = paras
        if add_comment:
            comments = c_dict[short_id]
            contents = paras + comments

        contents = contents[:100]

        write(short_id, title, contents, statements)


def write_split(name):
    file_path = os.path.join(data_path, "guardian", "claim", name)
    data = load_claim_annot(file_path)
    articles = load_commented_articles_opinion()

    long_url_indice = {}
    for a in articles:
        id, title, short_id, infos = a
        w = infos['webUrl']
        long_url_indice[w] = a


    train_num = int(len(data) * 0.7)
    valid_num = int(len(data) * 0.15)
    test_num = len(data) - train_num - valid_num

    names_d = {
        'train':[],
        'valid':[],
        'test':[],
    }

    for idx, entry in enumerate(data):
        webUrl = entry['url']
        article = long_url_indice[webUrl]
        id, title, short_id, infos = article
        paras = infos["paragraphs"]
        statements = entry['statements']
        fname = short_id.replace("/", "_") + ".story"
        if idx < train_num:
            names_d['train'].append(fname)
        elif idx < train_num+valid_num:
            names_d['valid'].append(fname)
        else:
            names_d['test'].append(fname)

    out_p = os.path.join(data_path, "guardian", "split.json")
    json.dump(names_d, open(out_p, "w"))

if __name__ == "__main__":
    write_guardian_claims("claim2.csv", True)
    #write_split("claim2.csv")

