import csv
import os
from typing import List, Dict

from arg.perspectives.load import claims_to_dict, get_all_claims
from cpath import output_path
from datastore.clueweb_helper import remove_duplicate
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
    claims = get_all_claims()
    claim_d = claims_to_dict(claims)

    keys = list(ranked_list.keys())
    keys.sort()
    num_doc_per_query = 10
    url_prefix = "http://localhost:36559/document?identifier="
    rows = []
    for query_id in keys[:10]:
        entries: List[SimpleRankedListEntry] = ranked_list[query_id]
        entries = entries[:num_doc_per_query * 3]
        doc_ids: List[str] = remove_duplicate(list([e.doc_id for e in entries]))
        claim = claim_d[int(query_id)]
        s = "{} : {}".format(query_id, claim)
        rows.append([Cell(s)])
        for doc_id in doc_ids[:num_doc_per_query]:
            url = url_prefix + doc_id
            s = "<a href=\"{}\">{}</a>".format(url, doc_id)
            rows.append([Cell(s)])

    html = HtmlVisualizer("claim_docs_urls.html")
    html.write_table(rows)


def write_csv(config):
    # select claims
    # load relevant documents
    # remove duplicate
    q_res_path = config['q_res_path']
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    claims = get_all_claims()
    claim_d = claims_to_dict(claims)

    keys = list(ranked_list.keys())
    keys.sort()
    num_doc_per_query = 10
    url_prefix = "http://gosford.cs.umass.edu:36559/document?identifier="
    rows = []

    header = ["claim"] + ["url{}".format(i) for i in range(1, num_doc_per_query+1)]
    rows.append(header)
    for query_id in keys[:10]:
        entries: List[SimpleRankedListEntry] = ranked_list[query_id]
        entries = entries[:num_doc_per_query * 3]
        doc_ids: List[str] = remove_duplicate(list([e.doc_id for e in entries]))
        claim = claim_d[int(query_id)]
        urls = []
        for doc_id in doc_ids[:num_doc_per_query]:
            url = url_prefix + doc_id
            urls.append(url)

        assert len(urls) == num_doc_per_query
        row = [claim] + urls
        rows.append(row)

    save_path = os.path.join(output_path, "claim10_train.csv")
    f = open(save_path, "w")

    csv_writer = csv.writer(f)
    csv_writer.writerows(rows)


if __name__ == "__main__":
    run_func_with_config(write_csv)

