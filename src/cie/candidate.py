import nltk
import os
from nltk import word_tokenize
import spacy
import en_core_web_sm
from path import data_path
nlp = en_core_web_sm.load()

def get_verbs(sent):
    tags = nltk.pos_tag(word_tokenize(sent))
    verbs = []
    for w, pos in tags:
        if "VB" in pos:
            verbs.append(w)
    return verbs

def get_nouns(sent):
    ne_list = []
    data = nlp(sent)
    for X in data.ents:
        # pprint((X.text, X.label_))
        text = str(X.text)
        ne_list.append(text.replace("\n", " "))
    return ne_list

def generate(all_sents):
    verbs_all = set()
    nouns_all = set()

    for sent in all_sents[:500]:
        verbs_all.update(get_verbs(sent))
        nouns_all.update(get_nouns(sent))

    return list(verbs_all), list(nouns_all)

def list_print(l, width):
    cnt = 0
    for item in l:
        print(item, end=" / ")
        cnt += 1
        if cnt == width:
            print()
            cnt = 0
    print()

if __name__ == "__main__":
    test_path = os.path.join(data_path, "cont_rel", "doc.txt")
    doc = open(test_path, "r").read()
    sents = nltk.sent_tokenize(doc)
    v_all, n_all = generate(sents)
    print("Verbs")
    list_print(v_all, 10)
    print("Nouns")
    list_print(n_all, 10)
