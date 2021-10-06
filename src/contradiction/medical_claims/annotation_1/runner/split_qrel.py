import sys


def main():
    qrel_path = sys.argv[1]

    val_split = list(range(13))
    f = open(qrel_path, "r")
    f_val = open(qrel_path + ".val", "w")
    f_test = open(qrel_path + ".test", "w")

    for line in f:
        qid = line.split()[0]
        data_id_str = qid.split("_")
        data_group_no = int(data_id_str[0])
        if data_group_no in val_split:
            f_val.write(line)
        else:
            f_test.write(line)


if __name__ == "__main__":
    main()