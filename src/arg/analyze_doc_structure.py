import pickle
from collections import Counter

import bs4
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from crawl.crawl_ukp_docs import get_topic_doc_save_path
from data_generator.argmining.document_stat import load_with_doc
from misc_lib import average

stop_words = set(stopwords.words("english"))


def load_topic_docs(topic):
    save_path = get_topic_doc_save_path(topic)
    data = pickle.load(open(save_path, "rb"))
    return data



class Sent:
    def __init__(self, raw_sent, source_elem, iv, sent_id):
        self.raw_sent = raw_sent
        self.sent_id = sent_id

        self.bow = Counter()
        for trigram in get_trigrams(raw_sent):
            self.bow[trigram] += 1

        self.source_elem = source_elem
        for trigram in self.bow:
            if trigram not in iv:
                iv[trigram] = list()
            iv[trigram].append(self)


def get_trigrams(sent):
    tokens = word_tokenize(sent)
    st = 0
    while st + 3 <= len(tokens):
        trigram = " ".join(tokens[st:st+2])
        yield trigram
        st += 1


def cover_count(counter_1, counter_2):
    s = []
    for key in counter_1:
        coverage = counter_2[key] / counter_1[key]
        s.append(coverage)

    return average(s)

def traverse(soup):
    if soup.name is not None:
        dom_dictionary = {}
        dom_dictionary['name'] = soup.name
        dom_dictionary['children'] = [ traverse(child) for child in soup.children if child.name is not None]
        return dom_dictionary


def print_selected(soup, indent):
    if soup.name is not None:
        if 'selected' in soup.attrs:
            line = '-' * indent + soup.name + " " + soup.attrs['selected']
            if soup.attrs['selected'] is not 'parent':
                line += ": " + soup.text.strip()[:200]
            print(line)
            for child in soup.children:
                print_selected(child, indent+2)
        elif soup.name in ["h1", "h2", "h3", "h4"] or 'title_like' in soup.attrs:
            line = '-' * indent + soup.name + " " + soup.text.strip()[:40]
            print(line)
            return


def mark_selected(soup):
    if soup.name is not None:
        child_mark = False
        for child in soup.children:
            r = mark_selected(child)
            child_mark = child_mark or r
        if 'selected' not in soup.attrs and child_mark:
            soup.attrs['selected'] = 'parent'
        return child_mark or ('selected' in soup.attrs)
    else:
        return False


# If node is strong return True
# If node is p and child is 'strong' mark as title_like
known_names = set()
def mark_title_like_strong(soup):
    if soup.name is not None:
        if soup.name == 'strong':
            return True
        n_strong = 0
        n_child = 0
        for child in soup.children:
            is_strong = mark_title_like_strong(child)
            if is_strong:
                n_strong += 1
            n_child += 1
        if n_strong == 1 and n_child == 1:
            soup.attrs['title_like'] = 'strong'
        return False
    else:
        return False


def match(content, labeld_data):
    soup = bs4.BeautifulSoup(content, features="html.parser")
    iv = dict()

    all_sent_from_docs = []
    for elem in soup.find_all('p'):
        text = elem.get_text()
        for sent in sent_tokenize(text):
            sent_id = len(all_sent_from_docs)
            sent = Sent(sent, elem, iv, sent_id)
            all_sent_from_docs.append(sent)
    n_fount = 0
    n_not_found = 0
    for entry in labeld_data:
        sent = entry['sentence']
        y = entry['annotation']
        trigrams = get_trigrams(sent)
        bow = Counter(trigrams)
        candidate = set()
        for term in bow:
            if term not in iv:
                continue

            for doc_sent in iv[term]:
                candidate.add(doc_sent)


        ranked_list = []
        for doc_sent in candidate:
            score = cover_count(bow, doc_sent.bow)
            ranked_list.append((score, doc_sent))

        ranked_list.sort(key=lambda x:x[0], reverse=True)
        #print(sent)


        if ranked_list and ranked_list[0][0] > 0.5 :
            n_fount += 1
            score, sent = ranked_list[0]
            if 'selected' not in sent.source_elem.attrs:
                sent.source_elem.attrs['selected'] = y
            else:
                sent.source_elem.attrs['selected'] += ", " + y
            #print(score, item.raw_sent)
        else:
            n_not_found += 1
            #print("{} candidates ".format(len(candidate)))
            #for e in ranked_list[:10]:
            #    score, item = e
            #   print(score, item.raw_sent)
    t = mark_selected(soup)
    mark_title_like_strong(soup)
    print(n_fount, n_not_found)
    print_selected(soup, 0)

def main():
    topic = "abortion"
    all_data = load_with_doc()
    topic_data = all_data[topic]

    data = load_topic_docs(topic)
    for url, content in data.items():
        entries = topic_data[url]
        print(url)
        match(content, entries)





if __name__ == "__main__":
    main()