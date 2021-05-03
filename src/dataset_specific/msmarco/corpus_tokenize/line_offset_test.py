from typing import List

from dataset_specific.msmarco.common import at_working_dir


def load_offset() -> List[int]:
    out_f = open(at_working_dir("msmarco-docs-offset.tsv"), "r", encoding="utf8")
    offset_list = []
    for line_idx, line in enumerate(out_f):
        idx, offset = line.split("\t")
        assert line_idx == int(idx)
        offset_list.append(int(offset))
    return offset_list


def main():
    f = open(at_working_dir("msmarco-docs.tsv"), "r", encoding="utf8")

    offset_list = load_offset()

    def read_line(line_no):
        f.seek(offset_list[line_no])
        return f.readline()

    for line_no in [10, 1000, 248152]:
        print("Line {}: {}".format(line_no, read_line(line_no)))


if __name__ == "__main__":
    main()