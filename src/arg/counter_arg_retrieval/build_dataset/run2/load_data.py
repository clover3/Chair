import csv
import os

from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from cpath import output_path


def load_my_run2_topics():
    csv_path = os.path.join(output_path, "ca_building", "run2", "written_ca_query.csv")
    reader = csv.reader(open(csv_path, "r"))

    qid = 1
    output = []
    for row_idx, row in enumerate(reader):
        if row_idx == 0:
            continue
        output.append(CAQuery(str(qid), row[0], row[1], row[2]))
        qid += 1
    return output


def main():
    load_my_run2_topics()


if __name__ == "__main__":
    main()
