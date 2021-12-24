import os
import random
from collections import Counter

from cache import save_to_pickle, load_pickle_from, load_from_pickle
from epath import job_man_dir


def contain_person(e):
    qid, query, spacy_tokens = e
    for entity in spacy_tokens.ents:
        if entity.label_ == "PERSON":
            return True
    return False


def get_head_tail(spacy_tokens):
    is_head = True
    head = []
    tail = []
    body = []
    for token in spacy_tokens:
        if token.ent_type_ == "PERSON":
            body.append(str(token))
            is_head = False
        else:
            if is_head:
                head.append(str(token))
            else:
                tail.append(str(token))
    return head, body, tail


def parse_query(e):
    qid, query, spacy_tokens = e
    head, body, tail = get_head_tail(spacy_tokens)
    func_spans = (head, tail)
    d = {
        'qid': qid,
        'query': query,
        'content_span': body,
        'functional_span': func_spans,
    }
    return d


def query_parsing_debug(num_jobs, split):
    qid_query_tokens_list = enum_qid_token_list(num_jobs, split)
    iter = filter(contain_person, qid_query_tokens_list)
    iter = map(parse_query, iter)

    counter = Counter()
    save_obj = []
    for parsed_d in iter:
        head, tail = parsed_d['functional_span']
        sig = " ".join(head) + " [ENTITY] " + " ".join(tail)
        save_obj.append(parsed_d)
        counter[sig] += 1

    print("Total of {} func_span".format(len(counter)))
    t = 5
    frequent_spnas = [k for (k, v) in counter.items() if v > t]
    print("{} spans appear more than {} times ".format(len(frequent_spnas), t))
    save_to_pickle(save_obj, "mmd_query_entity_parse_{}".format(split))


def enum_qid_token_list(num_jobs, split):
    for i in range(num_jobs):
        pickle_path = os.path.join(job_man_dir, "msmarco_spacy_query_parse_{}".format(split), "{}".format(i))
        if os.path.exists(pickle_path):
            l = load_pickle_from(pickle_path)
            print("{} loaded from {}".format(len(l), i))
            yield from l


def load_print_parse_results():
    split = "train"
    save_obj = load_from_pickle("mmd_query_entity_parse_{}".format(split))
    print("{} queries".format(len(save_obj)))
    random.shuffle(save_obj)
    counter = Counter()
    for parsed_d in save_obj:
        head, tail = parsed_d['functional_span']
        sig = " ".join(head) + " [ENTITY] " + " ".join(tail)
        counter[sig] += 1

    for sig, cnt in counter.most_common():
        print(sig, cnt)
        if cnt < 5:
            break

def show_entity_types(num_jobs, split):
    qid_query_tokens_list = enum_qid_token_list(num_jobs, split)

    counter = Counter()
    for e in qid_query_tokens_list:
        qid, query, spacy_tokens = e
        counter["ALL"] += 1
        for entity in spacy_tokens.ents:
            entity_type = str(entity.label_)
            if entity_type not in counter:
                print(entity_type)
            counter[entity_type] += 1

        if counter["ALL"] > 45000:
            break

    for k, v in counter.items():
        print(k, v)


def main():
    show_entity_types(1, "train")
    # query_parsing_debug(8, "train")
    # load_print_parse_results()


if __name__ == "__main__":
    main()
