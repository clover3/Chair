import json
from typing import List, Dict

from exec_lib import run_func_with_config
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from visualize.html_visual import Cell, HtmlVisualizer


def main(config):
    # select claims
    # load relevant documents
    # remove duplicate
    q_res_path = config['q_res_path']
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    query_text_d = json.load(open(config['query_text_d']))
    save_name = config['save_path']

    keys = list(ranked_list.keys())
    keys.sort()
    num_doc_per_query = 10
    url_prefix = "http://localhost:36559/document?identifier="
    rows = []
    for query_id in keys[:100]:
        entries: List[SimpleRankedListEntry] = ranked_list[query_id]
        entries = entries[:num_doc_per_query * 3]
        doc_ids: List[str] = list([e.doc_id for e in entries])
        query_text = query_text_d[query_id]
        s = "{} : {}".format(query_id, query_text)
        rows.append([Cell(s)])
        for doc_id in doc_ids[:num_doc_per_query]:
            url = url_prefix + doc_id
            s = "<a href=\"{}\">{}</a>".format(url, doc_id)
            rows.append([Cell(s)])

    html = HtmlVisualizer(save_name)
    html.write_table(rows)


if __name__ == "__main__":
    run_func_with_config(main)

