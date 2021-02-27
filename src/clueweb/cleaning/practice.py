import difflib
import json
from typing import Dict, Tuple

import nltk.tokenize
from bs4 import BeautifulSoup
from bs4.element import Comment

from cache import load_from_pickle
from cpath import at_output_dir
from list_lib import lmap


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'a']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def bs4_clean(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.body.get_text().lower()


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.body.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    tokens = [t.strip() for t in visible_texts]
    tokens = [t for t in tokens if t]
    return u" ".join(tokens).lower()


def print_diff(diff_list):
    prev_sign = None
    buffer = []
    for line in diff_list:
        sign = line[:2]
        text = line[2:]
        assert text.strip()
        if sign == prev_sign:
            buffer.append(text)
        else:
            if buffer:
                print("--------------")
                print("[{}]".format(sign) + " ".join(buffer))
                buffer = []
            buffer.append(text)

        prev_sign = sign


def main():
    clean_doc_sample: Dict[str, Tuple] = load_from_pickle("clean_clueweb_doc_sample")
    doc_json_list = lmap(json.loads, open(at_output_dir("clueweb", "doc_content_samples.json"), "r"))
    d = difflib.Differ()
    html_diff = difflib.HtmlDiff()

    clean_fn = text_from_html
    for doc_json in doc_json_list[1:]:
        html = doc_json['content']
        open(at_output_dir("visualize", "text.html"), "w", encoding="utf-8").write(html)
        doc_id = doc_json['id']
        title, cleaned_text_ref = clean_doc_sample[doc_id]
        cleaned_text = clean_fn(html)
        # print(cleaned_text_ref)
        # print(cleaned_text)

        tokens_ref = nltk.tokenize.wordpunct_tokenize(cleaned_text_ref)
        tokens = nltk.tokenize.wordpunct_tokenize(cleaned_text)
        print(" ".join(tokens_ref))
        print(" ".join(tokens))
        diff = d.compare(tokens_ref, tokens)
        break


if __name__ == "__main__":
    main()