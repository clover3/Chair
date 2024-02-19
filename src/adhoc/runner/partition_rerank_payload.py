import sys
import os


def main():
    input_path = sys.argv[1]
    n_per_file = int(sys.argv[2])
    file_name, file_extension = os.path.splitext(input_path)

    lines = open(input_path, "r").readlines()

    def is_qid_equal(line1, line2):
        qid1, pid, _, _ = line1.split("\t")
        qid2, pid, _, _ = line2.split("\t")
        return qid1 == qid2

    partition_no = 0
    cursor = 0
    while cursor < len(lines):
        st = cursor
        ed = st + n_per_file
        if ed < len(lines):
            if is_qid_equal(lines[ed-1], lines[ed]):
                raise ValueError()

        cur_block = lines[st: ed]
        save_file_name = f"{file_name}_{partition_no}{file_extension}"
        open(save_file_name, "w").writelines(cur_block)
        cursor = ed
        partition_no += 1


if __name__ == "__main__":
    main()