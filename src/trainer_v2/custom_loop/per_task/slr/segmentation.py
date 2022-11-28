from typing import List
from typing import TypeVar

import spacy
from spacy.tokens import Doc
from spacy.tokens import Span

SpacyToken = TypeVar('SpacyToken')


def spacy_segment(spacy_tokens: Doc) -> List[Span]:
    #  extract noun-phrases
    np_list: List[Span] = list(spacy_tokens.noun_chunks)
    #  convert noun-phrase to propositional phrase by tag IN
    ed_list = []
    span_st_ed = []
    last_end = 0
    for np in np_list:
        ed_list.append(np.end)
        st = last_end
        ed = np.end
        span_st_ed.append((st, ed))
        last_end = ed

    st, ed = span_st_ed[-1]
    span_st_ed[-1] = st, len(spacy_tokens)
    spans_to_use = []
    for st, ed in span_st_ed:
        spans_to_use.append(spacy_tokens[st:ed])
    return spans_to_use


def main():
    nlp = spacy.load("en_core_web_sm")
    sent = "Ada Lovelace was born in London"
    sent = "The book written by the guy who lived in London being occupied is sad."
    doc: Doc = nlp(sent)
    spacy_segment(doc)
    doc = nlp("A phrase with another phrase occurs.")
    chunks = list(doc.noun_chunks)
    assert len(chunks) == 2
    assert chunks[0].text == "A phrase"
    assert chunks[1].text == "another phrase"


if __name__ == "__main__":
    main()