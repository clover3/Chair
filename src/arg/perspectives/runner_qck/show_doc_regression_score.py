import argparse
import sys
from typing import List, Dict

from arg.qck.decl import qk_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from misc_lib import group_by
from tab_print import print_table
from tlm.estimator_output_reader import join_prediction_with_info

parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--prediction_path")
parser.add_argument("--info_path")


def get_regression_score(entry):
    return entry['logits'][0]


def main():
    args = parser.parse_args(sys.argv[1:])
    extract_qk_unit(args.info_path, args.prediction_path)


def extract_qk_unit(info_path, pred_path):
    info = load_combine_info_jsons(info_path, qk_convert_map, False)
    predictions = join_prediction_with_info(pred_path, info)
    grouped: Dict[str, List[Dict]] = group_by(predictions, lambda x: x['query'].query_id)

    rows = []
    for qid, entries in grouped.items():
        any_entry = entries[0]
        query = any_entry['query']
        rows.append([query.query_id, query.text])
        for entry in entries:
            row = [get_regression_score(entry),
                   entry['kdp'].doc_id,
                   entry['kdp'].passage_idx]
            rows.append(row)

    print_table(rows)



if __name__ == "__main__":
    main()
