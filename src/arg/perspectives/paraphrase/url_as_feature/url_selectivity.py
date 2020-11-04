from collections import Counter

from exec_lib import run_func_with_config


def load_doc_ids(in_path):
    doc_ids = set([l.strip() for l in open(in_path, "r").readlines()])
    return doc_ids


def get_url_host(url):
    tokens = url.split("//")
    tail = tokens[1]
    host = tail.split("/")[0]
    return host
###

def main(config):
    pos_doc_ids = load_doc_ids(config['doc_list_path'])
    neg_doc_ids = load_doc_ids(config['neg_doc_save_path'])

    f = open(config['url_mapping_path'], "r")
    url_d = {}
    for line in f:
        if line.strip():
            doc_id, url = line.split()
            url_d[doc_id] = get_url_host(url)

    def ids_to_urls(doc_ids):
        return list([url_d[doc_id] for doc_id in doc_ids if doc_id in url_d])
    pos_urls = ids_to_urls(pos_doc_ids)
    neg_urls = ids_to_urls(neg_doc_ids)

    pos_url_counts = Counter(pos_urls)
    neg_url_counts = Counter(neg_urls)

    pos_double = 0
    both = 0
    neg_double = 0

    for key in pos_url_counts:
        if pos_url_counts[key] > 1:
            pos_double += 1
        elif neg_url_counts[key] >= 1:
            both += 1

    for key in neg_url_counts:
        if neg_url_counts[key] > 1:
            neg_double += 1

    print("P(url again in Pos | Url in Pos)", pos_double / len(pos_urls))
    print("P(url in Neg| Url in Pos)", both / len(pos_urls))
    print("P(url again in Neg | Url in Neg)", neg_double / len(neg_urls))


if __name__ == "__main__":
    run_func_with_config(main)