import csv
import os
from typing import List

from arg.counter_arg_retrieval.build_dataset.judgments import JudgementEx
from arg.counter_arg_retrieval.build_dataset.judgments_helper import save_judgement_entries_from_ex
from arg.counter_arg_retrieval.build_dataset.run5.data_path_common import get_run5_query_set
from arg.counter_arg_retrieval.build_dataset.run5.judgments.get_judgments import get_judgments_todo
from cpath import output_path
from table_lib import read_csv_as_dict


def main():
    todo: List[JudgementEx] = get_judgments_todo()
    ca_tasks = get_run5_query_set()
    save_path = os.path.join(output_path, "ca_building", "run5", "annot_jobs.csv")
    save_judgement_entries_from_ex(todo,
                                   ca_tasks,
                                   save_path)

def augment_neg(neg_claim_path, annot_src_path, annot_save_path):
    insert_col_idx = 2
    new_col = "neg_claim"

    neg_claims = read_csv_as_dict(neg_claim_path)

    cid_to_neg_claim = {}
    for d in neg_claims:
        cid_to_neg_claim[d['cid']] = d['neg_claim']
    reader = csv.reader(open(annot_src_path, "r", encoding="utf-8"))
    out_table = []
    for g_idx, row in enumerate(reader):
        if g_idx == 0:
            head = row
            head = head[:insert_col_idx] + [new_col] + head[insert_col_idx:]
            out_table.append(head)
        else:
            qid = row[0]
            cid, _ = qid.split("_")
            neg_claim = cid_to_neg_claim[cid]
            row = row[:insert_col_idx] + [neg_claim] + row[insert_col_idx:]
            out_table.append(row)

    writer = csv.writer(open(annot_save_path, "w", newline="", encoding="utf-8"))
    writer.writerows(out_table)


def augment_neg_claims():
    neg_claim_path = os.path.join(output_path, "ca_building", "run5", "neg_claims.csv")
    annot_src_path = os.path.join(output_path, "ca_building", "run5", "annot_jobs.csv")
    annot_save_path = os.path.join(output_path, "ca_building", "run5", "annot_jobs_neg.csv")
    augment_neg(neg_claim_path, annot_src_path, annot_save_path)


if __name__ == "__main__":
    augment_neg_claims()
