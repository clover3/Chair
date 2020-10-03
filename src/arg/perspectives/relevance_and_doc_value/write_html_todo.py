import os

from arg.perspectives.doc_value_viewer.show_doc_value2 import load_train_claim_d
from cpath import output_path
from tab_print import tab_print
from visualize.html_visual import HtmlVisualizer, Cell


def load_claim_id_and_doc_id():
    f = open(os.path.join(output_path, "doc_ids_with_effect.txt"), "r")
    for line in f:
        try:
            claim_id, doc_id, score = line.split()
            yield claim_id, doc_id
        except ValueError:
            pass


def main():

    claim_d = load_train_claim_d()
    html = HtmlVisualizer("doc_relevance_and_value.html")
    rows = []
    data_id = 0
    for claim_id, doc_id in load_claim_id_and_doc_id():
        claim_text = claim_d[int(claim_id)]
        url = os.path.join(output_path, "pc_docs_html", doc_id + ".html")
        a = "<a href=\"{}\">url</a>".format(url)
        tab_print(data_id, claim_id, doc_id)
        row = [Cell(data_id), Cell(claim_id), Cell(claim_text), Cell(a)]
        rows.append(row)
        data_id += 1

    html.write_table(rows)



if __name__ == "__main__":
    main()