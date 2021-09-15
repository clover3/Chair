import csv
import os
from typing import Tuple, List

from misc_lib import exist_or_mkdir


def load_corpus(path) -> List[Tuple[str, str]]:
    f = open(path, "r", encoding="utf-8", errors='ignore')
    output = []
    for row in csv.reader(f):
        output.append((row[0], row[1]))
    return output


def escape_sentence(s):
    s = s.replace("\"", "&quot;").replace("\'", "&lsquo;")
    return " ".join(s.split())


def generate_5per_hit(input_csv_path, html_template_path,
                      html_out_dir,
                      link_out_path,
                      ):
    exist_or_mkdir(html_out_dir)
    csv_writer = csv.writer(open(link_out_path, "w", newline='', encoding="utf-8"))
    place_holder = ["dumny", "dummy"]

    html_template = open(html_template_path, "r").read()
    data = list(load_corpus(input_csv_path))

    csv_writer.writerow(["url1", "url2", "url3", "url4", "url5"])
    name_list = []
    for idx, (prem, hypo) in enumerate(data):
        replace_mapping = {
            'SuperWildCardMyIfThePremise': escape_sentence(prem),
            'SuperWildCardMyIfTheHypothesis': escape_sentence(hypo),
            "SuperWildCardMyIfID": str(idx)
        }
        new_html = html_template
        for place_holder, value in replace_mapping.items():
            new_html = new_html.replace(place_holder, value)

        new_name = "{}.html".format(idx)
        open(os.path.join(html_out_dir, new_name), "w", errors="ignore").write(new_html)
        name_list.append(new_name)

    n_task = int(len(data) / 5)

    for i in range(n_task):
        b = i * 5
        entry = [name_list[b], name_list[ b +1], name_list[ b +2], name_list[ b +3], name_list[ b +4]]
        csv_writer.writerow(entry)


def generate_hit(input_csv_path, html_template_path,
                      html_out_dir,
                      link_out_path,
                      url_prefix,
                      ):
    exist_or_mkdir(html_out_dir)
    csv_writer = csv.writer(open(link_out_path, "w", newline='', encoding="utf-8"))

    html_template = open(html_template_path, "r").read()
    data = list(load_corpus(input_csv_path))
    csv_writer.writerow(["url"])
    for idx, (prem, hypo) in enumerate(data):
        replace_mapping = {
            'SuperWildCardMyIfThePremise': escape_sentence(prem),
            'SuperWildCardMyIfTheHypothesis': escape_sentence(hypo),
            "SuperWildCardMyIfID": str(idx)
        }
        new_html = html_template
        for place_holder, value in replace_mapping.items():
            new_html = new_html.replace(place_holder, value)

        new_name = "{}.html".format(idx)
        open(os.path.join(html_out_dir, new_name), "w", errors="ignore").write(new_html)

        url = url_prefix + new_name
        entry = [url]
        csv_writer.writerow(entry)

