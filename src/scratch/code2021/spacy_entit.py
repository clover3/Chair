import spacy

from cache import load_from_pickle
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo


def enum_entity_from_query():
    nlp = spacy.load("en_core_web_sm")
    qtype_entries, query_info_dict = load_from_pickle("run_analysis_dyn_qtype")
    seen = set()
    skip = 0
    for e_idx, e in enumerate(qtype_entries):
        info: QueryInfo = query_info_dict[e.qid]
        q_rep = " ".join(info.out_s_list)
        query = info.query
        if query not in seen:
            doc = nlp(query)
            appropriate = False
            for e in doc.ents:
                if e.text == info.content_span:
                    print("Skipped {} queries".format(skip))
                    skip = 0
                    print(q_rep)
                    print(e.text, e.label_, e.kb_id_)
                    appropriate = True

            if not appropriate:
                skip += 1
            seen.add(query)


def split_by_entity():
    nlp = spacy.load("en_core_web_sm")
    sent = "john allen killed darlen?"
    doc = nlp(sent)
    head, tail = get_head_tail(doc)

    print((head, tail))


def get_head_tail(spacy_tokens):
    is_head = True
    head = []
    tail = []
    for token in spacy_tokens:
        if token.ent_type_ == "PERSON":
            is_head = False
        else:
            if is_head:
                head.append(str(token))
            else:
                tail.append(str(token))
    return head, tail


def main():
    split_by_entity()


if __name__ == "__main__":
    main()