import csv
import os
from typing import NamedTuple

from cpath import output_path


class CAQuery(NamedTuple):
    qid: str
    claim: str
    perspective: str
    ca_query: str


def load_run2_topics():
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
    load_run2_topics()


if __name__ == "__main__":
    main()