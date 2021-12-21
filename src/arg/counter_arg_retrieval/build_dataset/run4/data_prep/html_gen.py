import csv
import os
from typing import List, Dict

from cpath import output_path
from misc_lib import exist_or_mkdir


def load_csv_dict_format(file_path) -> List[Dict]:
    f = open(file_path, "r", encoding="utf-8")
    outputs: List[Dict] = []
    head = []
    for idx, row in enumerate(csv.reader(f)):
        if idx == 0:
            head = row
        else:
            d = {}
            for column, value in zip(head, row):
                d[column] = value
            outputs.append(d)
    return outputs


def generate_hit(input_csv_path,
                 html_template_path,
                 html_out_dir,
                 ):
    exist_or_mkdir(html_out_dir)
    todo = load_csv_dict_format(input_csv_path)
    html_template = open(html_template_path, "r").read()
    for idx, entry in enumerate(todo):
        replace_mapping = {
            'PlaceHolderForPremise': entry['p_text'],
            'PlaceHolderForPStance': entry['p_stance'],
            'PlaceHolderForConclusion': entry['c_text'],
            'PlaceHolderForCStance': entry['c_stance'],
            'PlaceHolderForEntity': entry['entity'],
            "PlaceHolderForTaskNo": str(idx),
            "PlaceHolderForPassage": entry['passage']
        }
        new_html = html_template
        for place_holder, value in replace_mapping.items():
            new_html = new_html.replace(place_holder, value)

        new_name = "{}.html".format(idx)
        open(os.path.join(html_out_dir, new_name), "w", errors="ignore").write(new_html)


def main():
    html_template_path = os.path.join(output_path, "ca_building", "html", "annot_template.html")
    html_out_root = os.path.join(output_path, "ca_building", "html_output")
    exist_or_mkdir(html_out_root)

    input_csv_path = os.path.join(output_path, "ca_building", "run4", "annot_jobs.csv")
    generate_hit(input_csv_path, html_template_path, html_out_root)


if __name__ == "__main__":
    main()
