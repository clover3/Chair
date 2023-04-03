from cpath import output_path, data_path
from dataset_specific.mnli.mnli_reader import MNLIReader
from misc_lib import path_join


def main():
    save_root = path_join(data_path, "nli", "mnli_line")


    def do_for_split(split):
        reader = MNLIReader()
        out_f_d = {}
        for role in ["s1", "s2", "labels"]:
            save_path = path_join(save_root, f"{role}.{split}")
            f = open(save_path, "w", encoding="utf-8")
            out_f_d[role] = f

        for item in reader.load_split(split):
            mapping = {
                "s1": item.premise,
                "s2": item.hypothesis,
                "labels": item.label,
            }
            for role, text in mapping.items():
                out_f_d[role].write(text + "\n")


    for split in ["train", "dev"]:
        do_for_split(split)


if __name__ == "__main__":
    main()