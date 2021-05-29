import os

from contradiction.annotation.annotation_html_gen import generate_5per_hit
from cpath import data_path, at_output_dir


def main():
    html_template_path = os.path.join(data_path, "med_contradiction", "annotation", "annotation_template.html")
    input_csv_path = at_output_dir("alamri_pilot", "pilot_pairs.csv")
    html_out_dir = at_output_dir("alamri_pilot", "pilot_pairs_html")
    csv_link_output = at_output_dir("alamri_pilot", "pilot_links.csv")

    generate_5per_hit(input_csv_path, html_template_path, html_out_dir, csv_link_output)


if __name__ == "__main__":
    main()
