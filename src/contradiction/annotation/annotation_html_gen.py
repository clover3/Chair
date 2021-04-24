import csv
import os

from cpath import at_output_dir, data_path
from misc_lib import exist_or_mkdir


def load_corpus(path):
    f = open(path, "r", encoding="utf-8", errors='ignore')
    for row in csv.reader(f):
        yield row[0], row[1]


def escape_sentence(s):
    s = s.replace("\"", "&quot;").replace("\'", "&lsquo;")
    return " ".join(s.split())


def generate(input_csv_path, html_template_path,
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


def main():
    html_template_path = os.path.join(data_path, "med_contradiction", "annotation", "annotation_template.html")
    input_csv_path = at_output_dir("alamri_pilot", "pilot_pairs.csv")
    html_out_dir = at_output_dir("alamri_pilot", "pilot_pairs_html")
    csv_link_output = at_output_dir("alamri_pilot", "pilot_links.csv")

    generate(input_csv_path, html_template_path, html_out_dir, csv_link_output)


if __name__ == "__main__":
    main()