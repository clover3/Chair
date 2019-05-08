from path import data_path
import os
import json
from misc_lib import *
from collections import Counter
import pickle
from nltk.tokenize import wordpunct_tokenize
import numpy as np
scope_dir = os.path.join(data_path, "guardian")
comments_dir = os.path.join(scope_dir, "comments")
topic_dir = os.path.join(scope_dir, "topic")


def save_local_pickle(obj, name):
    pickle.dump(obj, open(os.path.join(data_path, "guardian", name + ".pickle"), "wb"))

def load_local_pickle(name):
    return pickle.load(open(os.path.join(data_path, "guardian", name + ".pickle"), "rb"))

def get_comments_text(file_path):
    comment_list = parse_comments(file_path)
    text_list = []
    for entry in comment_list:
        text_list.append(entry[1])
    return text_list


def parse_comments(file_path):
    comment_list = []

    f = open(file_path, encoding='utf-8')
    j = json.load(f)
    try:
        discussion = j['discussion']
        current_page = j['currentPage']
        discussion_id = discussion['key']
        web_url = discussion['webUrl']
        comment_count = j['discussion']['commentCount']
        comments = j['discussion']['comments']
        for comment in comments:
            comment_id = comment['id']
            comment_body = comment['body']
            comment_list.append((comment_id, comment_body))
            if 'responses' in comment:
                for response in comment['responses']:
                    res_body = response['body']
                    res_id = response['id']
                    comment_target = int(response['responseTo']['commentId'])
                    comment_list.append((res_id, res_body, comment_target))
    except KeyError as e:
        print(file_path)
    return comment_list


def discussion_stats():
    f = []
    for (dirpath, dirnames, filenames) in os.walk(comments_dir):
        f.extend(filenames)
        break

    count = Counter()
    for filename in f:
        full_path = os.path.join(comments_dir, filename)
        short_id = filename[:-len(".json")].replace("_", "/")
        comments = get_comments_text(full_path)
        count[len(comments)] += 1

    for key in count:
        print(key, count[key])

def get_reactions():
    return pickle.load(open(os.path.join(data_path, "guardian", "r_1.pickle"), "rb"))

def dispute_scores():
    topics = get_topics()
    reactions = get_reactions()

    topic_dicsussion_dict = {}
    entries = []
    for topic in topics:
        articles = get_topic_articles(topic)
        rank = 0
        sum_score = 0
        no_reaction =0
        for id, short_url, text in articles:
            has_reaction = short_url in reactions
            if has_reaction:
                res = reactions[short_url]
                sum_score += res
            else:
                res = "No reaction"
                no_reaction += 1
            rank += 1
        yes_reaction = len(articles) - no_reaction
        rel_score = sum_score / yes_reaction if yes_reaction > 0 else -1
        entries.append((topic, sum_score, no_reaction, rel_score))

        topic_dicsussion_dict[topic] = list([a[1] for a in articles])
    save_local_pickle(topic_dicsussion_dict, "topic_dicussions")


    entries.sort(key=lambda x:x[1], reverse=True)

    def get_rank(topic):
        for rank, e in enumerate(entries):
            if e[0] == topic:
                return rank
        return "Not found"

    print("Rank by sum_score")
    for e in entries[:20]:
        print(e[0])

    print("rank of scientology : ", get_rank("scientology"))
    print("rank of abortion : ", get_rank("abortion"))

    print("Rank by rel_score")
    entries.sort(key=lambda x: x[3], reverse=True)
    for e in entries[:20]:
        print(e[0])

    print("rank of scientology : ", get_rank("scientology"))
    print("rank of abortion : ", get_rank("abortion"))



def get_discussions(word):
    NotImplemented

def encode_discussions():
    word2idx = pickle.load(open(os.path.join(data_path, "guardian", "word2idx"), "rb"))
    f = []
    for (dirpath, dirnames, filenames) in os.walk(comments_dir):
        f.extend(filenames)
        break
    oov_set = set()
    result = []
    OOV = 1
    ticker = TimeEstimator(len(f), sample_size=100)

    def reform(indices):
        indices = indices[:100]
        indices = indices + (100 - len(indices)) * [0]
        return np.array(indices)

    for filename in f:
        full_path = os.path.join(comments_dir, filename)
        short_id = filename[:-len(".json")].replace("_", "/")
        comments = get_comments_text(full_path)[:100]

        enc_comments = []
        for text in comments:
            indices = []
            for token in wordpunct_tokenize(text):
                token = token.lower()
                if token in word2idx:
                    indices.append(word2idx[token])
                else:
                    indices.append(OOV)
                    oov_set.add(token)

            enc_comments.append(reform(indices))

        if enc_comments:
            entry = (short_id, None, np.stack(enc_comments))
            result.append(entry)
        ticker.tick()
    pickle.dump(result, open(os.path.join(data_path, "guardian", "code_comments.pickle"), "wb"))

def load_article(path):
    f = open(path, encoding='utf-8')
    j = json.load(f)
    articles = []
    for j_article in j['response']['results']:
        id = j_article['id']
        body_text = j_article['fields']['bodyText']
        short_url = j_article['fields']['shortUrl']
        short_id = short_url[-len("/p/ap83f"):]
        articles.append((id, short_id, body_text))
    return articles


def get_topics():
    topics = []
    for (dirpath, dirnames, filenames) in os.walk(topic_dir):
        topics.extend(dirnames)
        break
    return topics

def get_topic_articles(topic):
    json_path = os.path.join(topic_dir, topic, "1.json")
    return load_article(json_path)


def encode_docs():
    topics = get_topics()
    all_articles = []
    for topic in topics:
        articles = get_topic_articles(topic)
        all_articles.extend(articles)

    enc_entries = []
    ticker = TimeEstimator(len(all_articles), sample_size=100)
    for id, short_url, text in all_articles:
        tokens = wordpunct_tokenize(text)
        enc_text = list([t.lower() for t in tokens])
        enc_entries.append((short_url, enc_text))
        ticker.tick()

    f = open(os.path.join(data_path, "guardian", "code_articles.pickle"), "wb")
    pickle.dump(enc_entries, f)

def get_disagree_judge(discussion_id):
    NotImplemented


def estimate_unigram_dispute_potency():
    word_list = NotImplemented

    for word in word_list:
        discussion_list = get_discussions(word)

        lmap(get_disagree_judge, discussion_list)



if __name__ == "__main__":
    dispute_scores()
