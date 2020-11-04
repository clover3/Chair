


from typing import Set

# TODO given a document list, extract urls from galago
from exec_lib import run_func_with_config
from galagos.parse import get_doc_ids_from_ranked_list_path


def parse_line(line):
    offset = len("clueweb12-0000tw-00-00000")
    doc_id = line[:offset]
    url = line[offset+2 :]
    return doc_id, url


def main(config):
    all_doc_set : Set[str] = load_doc_ids_of_interest(config)
    f = open(config['url_mapping_path'], "r")


    result = []
    for idx, line in enumerate(f):
        doc_id, url = parse_line(line)

        if doc_id in all_doc_set:
            result.append((doc_id, url))

        if idx % (1000 * 100) == 0:

            print("cursor at {} found {}".format(idx, len(result)))

    fout = open(config['save_path'], "w")

    for doc_id, url in result:
        fout.write("{}\t{}\n".format(doc_id, url))


def load_doc_ids_of_interest(config):
    pos_doc_list_path = config['doc_list_path']
    q_res_path = config['q_res_path']
    pos_doc_ids = set([l.strip() for l in open(pos_doc_list_path, "r").readlines()])
    all_doc_list = get_doc_ids_from_ranked_list_path(q_res_path)
    neg_docs_ids = list([d for d in all_doc_list if d not in pos_doc_ids])
    sel_len = len(pos_doc_ids) * 5
    neg_docs_ids = neg_docs_ids[:sel_len]
    f = open(config['neg_doc_save_path'], "w")
    for doc_id in neg_docs_ids:
        f.write(str(doc_id) + "\n")
    pos_doc_ids.update(neg_docs_ids)
    return pos_doc_ids


if __name__ == "__main__":
    run_func_with_config(main)