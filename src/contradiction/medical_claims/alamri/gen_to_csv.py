import csv
import os

from contradiction.medical_claims.alamri.pilot_annotation import enum_true_instance
from cpath import output_path, at_output_dir
from list_lib import foreach
from misc_lib import DataIDManager, exist_or_mkdir


def main():
    exist_or_mkdir(os.path.join(output_path, "alamri_tfrecord"))

    data_id_manager = DataIDManager()
    entries = []
    for claim1, claim2 in enum_true_instance():
        entries.append((claim1.text, claim2.text))

    save_path = at_output_dir("alamri_pilot", "true_pairs_all.csv")
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, entries)


if __name__ == "__main__":
    main()
