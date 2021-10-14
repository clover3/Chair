import os

from contradiction.medical_claims.annotation_1.load_data import load_dev_sents
from cpath import at_output_dir
from misc_lib import exist_or_mkdir


def main():
    output = load_dev_sents()
    output_root = at_output_dir("alamri_annotation1", "sentences")
    exist_or_mkdir(output_root)

    for group_no, sents in output:
        save_path = os.path.join(output_root, "{}.tsv".format(group_no))
        f = open(save_path, "w")
        for s in sents:
            f.write(s + "\n")


if __name__ == "__main__":
    main()
