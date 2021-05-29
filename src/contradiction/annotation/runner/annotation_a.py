import os

from contradiction.annotation.annotation_html_gen import generate_hit
from cpath import data_path, at_output_dir, output_path
from misc_lib import exist_or_mkdir


def main():
    num_review = 24
    html_template_path = os.path.join(data_path, "med_contradiction", "annotation", "annotation_template.html")

    html_out_root = at_output_dir("alamri_annotation1", "html")
    exist_or_mkdir(html_out_root)

    for i in range(1, 1+num_review):
        input_csv_path = os.path.join(output_path, "alamri_annotation1", "grouped_pairs", "{}.csv".format(i))
        csv_link_output = os.path.join(output_path, "alamri_annotation1", "links", "{}.csv".format(i))
        html_out_dir = os.path.join(html_out_root, str(i))
        url_prefix = "https://ecc.neocities.org/{}/".format(i)
        generate_hit(input_csv_path, html_template_path, html_out_dir, csv_link_output, url_prefix)


if __name__ == "__main__":
    main()
