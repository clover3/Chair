import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import path
from arg.predictor import Predictor
from collections import Counter, defaultdict
from misc_lib import *
from nltk.tokenize import sent_tokenize
import tensorflow as tf
from cache import *


def get_relevant_docs(topic):
    dir_path = os.path.join(path.data_path, "arg", "plain")
    return load_dir_docs(os.path.join(dir_path, topic))

def load_dir_docs(dir_path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        f.extend(filenames)
        break

    result = []
    for filename in f:
        file_path = os.path.join(dir_path, filename)
        content = open(file_path, "r").read()
        if len(content) < 1:
            #print("Broken file : ", filename)
            continue
        result.append(content)

    return result

def get_keywords(topic):
    s = {"abortion": "abortion, woman, choice",
     "cloning": "cloning , research ",
     "death_penalty": "death penalty , punishment , execution , justice ",
     "gun_control": "gun control , rifles , firearms ",
     "marijuana_legalization": "marijuana legalization , cannabis , drug ",
     "minimum_wage": "minimum wage , labor  , worker ",
     "nuclear_energy": "nuclear energy , power , plant ",
     "school_uniforms": "school uniforms"}[topic]
    exptended_topic = list([t.strip() for t in s.split(",")])
    return exptended_topic[1:]


def check_stances(target_topic):
    #target_topic = "death_penalty"
    #keywords = ["punishment", "execution", "justice"]
    keywords = get_keywords(target_topic)
    predictor = Predictor(target_topic)
    def get_topic_stance(sents, target_topic):
        return predictor.predict(target_topic, sents)

    docs = get_relevant_docs(target_topic)[:100]

    window_size = [-3, 1] # inclusive
    #window_size = [0,0]
    def window(center_loc, list_len):
        start = max(0, center_loc + window_size[0])
        end = min(list_len-1, center_loc + window_size[1])
        return start, end+1

    def summarize_stance(list_stance):
        assert len(list_stance) > 0
        stance_count = Counter()
        for s in list_stance:
            stance_count[s] += 1
        if stance_count[1] > 0 and stance_count[2] > 0:
            return 3

        for stance in [1,2]:
            if stance_count[stance] > 0:
                return stance

        return 0

    line_split = sent_tokenize
    sents_list = lmap(line_split, docs)
    topic_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, target_topic), sents_list, False)
    """
    for doc_idx, doc in enumerate(docs):
        print("Doc #", doc_idx)
        sents = line_split(doc)
        for i, sent in enumerate(sents):
            print(topic_stances_list[doc_idx][i], sent)
    """

    for word in keywords:
        #print("Keyword = ", word)
        corr_mat = Counter()
        word_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, word), sents_list, False)
        for doc_idx, doc in enumerate(docs):
            sents = line_split(doc)
            num_sents = len(sents)
            if num_sents < 1:
                print("Skip doc #{}".format(doc_idx))
                continue

            topic_stances = topic_stances_list[doc_idx]
            word_stance = word_stances_list[doc_idx]

            assert len(topic_stances) > 0
            assert len(word_stance) > 0

            for i, sent in enumerate(sents):
                st, ed = window(i, num_sents)
                assert st < ed
                A_stance = summarize_stance(topic_stances[st:ed])
                B_stance = word_stance[i] #summarize_stance(word_stance[st:ed])
                corr_mat[(A_stance, B_stance)] += 1

        print("{}\t".format(word), end="")
        for j in range(0, 4):
            print("{}".format(j), end="\t")
        print("")
        for i in range(0,4):
            print("A={}".format(i), end="\t")
            for j in range(0,4):
                print("{}\t".format(corr_mat[i,j]), end="")
            print("")


def context_viewer(target_topic):
    docs = get_relevant_docs(target_topic)[:100]
    predictor = Predictor(target_topic)

    def get_topic_stance(sents, target_topic):
        return predictor.predict(target_topic, sents)

    window_size = [-3, 1] # inclusive
    #window_size = [0,0]
    def window(center_loc, list_len):
        start = max(0, center_loc + window_size[0])
        end = min(list_len-1, center_loc + window_size[1])
        return start, end+1

    line_split = sent_tokenize
    sents_list = lmap(line_split, docs)

    #topic_stances_list = load_from_pickle("stance_{}_rel.pickle".format(target_topic))
    topic_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, target_topic), sents_list, False)
    save_to_pickle(topic_stances_list, "stance_{}_rel.pickle".format(target_topic))

    def summarize_stance(list_stance):
        assert len(list_stance) > 0
        stance_count = Counter()
        for s in list_stance:
            stance_count[s] += 1
        if stance_count[1] > 0 and stance_count[2] > 0:
            return 3

        for stance in [1,2]:
            if stance_count[stance] > 0:
                return stance

        return 0

    def contains(sents, query):
        return query in " ".join(sents)
    count = Counter()
    for doc_idx, doc in enumerate(docs):
        sents = line_split(doc)
        num_sents = len(sents)
        if num_sents < 1:
            print("Skip doc #{}".format(doc_idx))
            continue

        topic_stances = topic_stances_list[doc_idx]
        for i, sent in enumerate(sents):
            st, ed = window(i, num_sents)
            A_stance = summarize_stance(topic_stances[st:ed])

            if A_stance in [1,2]:
                print("-------------")
                for j in range(st, ed):
                    print(topic_stances[j], sents[j])
                print("-------------")


def stance_pattern(target_topic, cheat_topic):
    docs = get_relevant_docs(target_topic)[:100]
    predictor = Predictor(target_topic, True, cheat_topic)

    def get_topic_stance(sents, target_topic):
        return predictor.predict(target_topic, sents)

    line_split = sent_tokenize
    sents_list = lmap(line_split, docs)

    f_html = open(os.path.join(path.output_path, "visualize", "stance_{}_cheat.html".format(target_topic)), "w")
    f_html.write("<html><head>\n")

    tooptip_style = open(os.path.join(path.data_path,"html", "tooltip")).read()
    f_html.write(tooptip_style)
    f_html.write("</head>\n")
    f_html.write("<h4>{}<h4>\n".format(target_topic))
    topic_stances_list = load_from_pickle("stance_{}_rel.pickle".format(target_topic))
    #topic_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, target_topic), sents_list, False)
    #save_to_pickle(topic_stances_list, "stance_{}_cheat.pickle".format(target_topic))

    for doc_idx, doc in enumerate(docs):
        sents = line_split(doc)
        num_sents = len(sents)
        if num_sents < 1:
            print("Skip doc #{}".format(doc_idx))
            continue



        topic_stances = topic_stances_list[doc_idx]
        count = Counter(topic_stances)

        p1 =count[1] / len(topic_stances)
        p2 = count[2] / len(topic_stances)
        f_html.write("<div>")
        f_html.write("<span>{0:.2f},{1:.2f}&nbsp;&nbsp;</span>".format(p1, p2))

        for i, stance in enumerate(topic_stances):
            tag = "<span class=\"tooltip\">{}\
            <span class=\"tooltiptext\">{}</span>\
            </span>".format(stance, sents[i])

            f_html.write(tag + "\n")
        f_html.write("</div>")
        f_html.write("<br>")
        print(topic_stances)
    f_html.write("\n</html>")

def pseudo_label(target_topic):
    docs = get_relevant_docs(target_topic)[:100]
    window_size = [-3, 1] # inclusive
    #window_size = [0,0]
    def window(center_loc, list_len):
        start = max(0, center_loc + window_size[0])
        end = min(list_len-1, center_loc + window_size[1])
        return start, end+1

    line_split = sent_tokenize
    sents_list = lmap(line_split, docs)

    #topic_stances_list = load_from_pickle("stance_abortion_rel.pickle")

    def get_topic_stance(sents, target_topic):
        return predictor.predict(target_topic, sents)

    predictor = Predictor(target_topic)
    topic_stances_list = flat_apply_stack(lambda x: get_topic_stance(x, target_topic), sents_list, False)

    def summarize_stance(list_stance):
        assert len(list_stance) > 0
        stance_count = Counter()
        for s in list_stance:
            stance_count[s] += 1
        if stance_count[1] > 0 and stance_count[2] > 0:
            return 3

        for stance in [1,2]:
            if stance_count[stance] > 0:
                return stance

        return 0

    pseudo_stances = defaultdict(list)

    for doc_idx, doc in enumerate(docs):
        sents = line_split(doc)
        num_sents = len(sents)
        if num_sents < 1:
            print("Skip doc #{}".format(doc_idx))
            continue

        topic_stances = topic_stances_list[doc_idx]
        for i, sent in enumerate(sents):
            st, ed = window(i, num_sents)
            A_stance = summarize_stance(topic_stances[st:ed])
            pseudo_stances[A_stance].append(sent)

    save_to_pickle(pseudo_stances, "{}_pseudo".format(topic))


if __name__ == "__main__":
    all_topics = ["abortion", "cloning", "death_penalty", "gun_control",
                  "marijuana_legalization", "minimum_wage", "nuclear_energy"]
    for topic in all_topics:
        print("-------------------")
        print(topic)
        tf.reset_default_graph()
        cheat_topic = "cloning" if topic != "cloning" else "death_penalty"
        stance_pattern(topic, cheat_topic)
