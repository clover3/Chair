from typing import List
from typing import TypeVar

import spacy
from spacy.tokens import Doc
from spacy.tokens import Span

from list_lib import flatten

SpacyToken = TypeVar('SpacyToken')


def spacy_segment(spacy_tokens: Doc) -> List[Span]:
    #  extract noun-phrases
    np_list: List[Span] = list(spacy_tokens.noun_chunks)
    #  convert noun-phrase to propositional phrase by tag IN
    pp_list: List[Span] = []
    remaining_np_list: List[Span] = []
    for np in np_list:
        exist_pp = np.root.head.tag_ == 'IN'
        if exist_pp:
            head_idx = np.root.head.i
            st = min(np.start, head_idx)
            ed = max(np.end, head_idx+1)
            pp = spacy_tokens[st:ed]
            pp_list.append(pp)
        else:
            remaining_np_list.append(np)

    #  verb + Preposition. (RP)
    # These VP are not precisely verb phrases, real verb phrases should have objects
    # vp_list may have overlap with np_list
    vp_list = []
    for token in spacy_tokens:
        if token.pos_ == "VERB":
            exist_pp = token.head.tag_ == "RP"
            if exist_pp:
                st = min(token.i, token.head.i)
                ed = max(token.i + 1, token.head.i + 1)
                verb_phrase = spacy_tokens[st:ed]
                vp_list.append(verb_phrase)
            else:
                vp_list.append(spacy_tokens[token.i: token.i+1])

    def span_iter_to_str(s_list):
        return "[" + " | ".join(list(map(str, s_list))) + "]"

    marked = set()
    for phrase in pp_list + remaining_np_list + vp_list:
        for token in phrase:
            marked.add(token.i)
    # print(marked)
    def covered(token: SpacyToken) -> bool:
        return token.i in marked

    def is_open_word(token) -> bool:
        return token.pos_ in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]

    #  keep remaining Open-class words
    #  Discard remaining.
    other_segments = []
    skip_words = []
    for token in spacy_tokens:
        if not covered(token):
            cur_token_span = spacy_tokens[token.i: token.i + 1]
            if is_open_word(token):
                other_segments.append(cur_token_span)
            else:
                skip_words.append(cur_token_span)

    # print("np / pp / vp / other")
    spans_to_use = [remaining_np_list, pp_list, vp_list, other_segments]
    # print(" / ".join(map(span_iter_to_str, spans_to_use)))
    number_of_tokens_in_span_to_use = 0
    for s_list in spans_to_use:
        for s in s_list:
            number_of_tokens_in_span_to_use += len(s)
        # number_of_tokens_in_span_to_use += sum(map(len, s_list))

    return list(flatten(spans_to_use))


def main():
    nlp = spacy.load("en_core_web_sm")
    doc: Doc = nlp("Ada Lovelace was born in London")
    some_token = doc[1]
    print(some_token.subtree, type(some_token.subtree))
    spacy_segment(doc)
    doc = nlp("A phrase with another phrase occurs.")
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text == "A phrase"
    assert chunks[1].text == "another phrase"


if __name__ == "__main__":
    main()