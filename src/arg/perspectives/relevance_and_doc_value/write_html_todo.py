import os
from typing import List

from arg.qck.decl import QKUnit
from cache import load_from_pickle
from cpath import output_path
from visualize.html_visual import HtmlVisualizer, Cell


def load_claim_id_and_doc_id():
    f = open(os.path.join(output_path, "doc_ids_with_effect.txt"), "r")
    for line in f:
        try:
            claim_id, doc_id, score = line.split()
            yield claim_id, doc_id
        except ValueError:
            pass


def load_qk() -> List[QKUnit]:
    qk_list = []
    for split in ["train"]:
        s = "pc_qk100_{}".format(split)
        qk_list.extend(load_from_pickle(s))
    return qk_list


def main():
    #claim_d = load_train_claim_d()
    html = HtmlVisualizer("doc_relevance_and_value.html")
    rows = []
    data_id = 0
    for query, k_list in load_qk():
        claim_id = query.query_id
        claim_text = query.text
        for k in k_list[:10]:
            doc_id = k.doc_id
            url = os.path.join(output_path, "pc_docs_html", doc_id + ".html")
            a = "<a href=\"{}\">url</a>".format(url)
            #tab_print(data_id, claim_id, doc_id)
            row = [Cell(data_id), Cell(claim_id), Cell(claim_text), Cell(a)]
            rows.append(row)
        data_id += 1

    html.write_table(rows)


if __name__ == "__main__":
    main()