import csv
from collections import Counter
from typing import Dict

from contradiction.medical_claims.annotation_1.reject_list import get_worker_list_to_reject


def main():
    # input_path = "C:\\work\\Code\\Chair\\output\\alamri_annotation1\\batch_results\\Batch_4516581_batch_results.csv"
    output_path = "C:\\work\\Code\\Chair\\output\\alamri_annotation1\\batch_results\\Batch_4516581_batch_to_reject.csv"
    #
    input_path = "C:\\work\\Code\\Chair\\output\\alamri_annotation1\\batch_results\\Batch_4500348_batch_results.csv"
    f = open(input_path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)
    head = list(data[0])
    row_idx_d: Dict[str, int] = {}
    workers = get_worker_list_to_reject()

    reason_reject = "Unapproved access"

    for idx, column_name in enumerate(head):
        row_idx_d[column_name] = idx

    counter = Counter()
    future_todo = []
    for row in data[1:]:
        worker_id = row[row_idx_d['WorkerId']]

        status = row[row_idx_d['AssignmentStatus']]
        if worker_id in workers:
            counter['n_bad_assn'] += 1
            if status == 'Approved':
                future_todo.append(row[row_idx_d['Input.url']])
                counter['n_bad_assn Approved'] += 1
            else:
                counter['n_bad_assn not approved'] += 1

                idx_reject = row_idx_d['Reject']
                while len(row) <= idx_reject:
                    row.append("")
                row[idx_reject] = reason_reject
    #
    # f_out = open(output_path, "w", encoding="utf-8", newline="")
    # csv_writer = csv.writer(f_out)
    # for row in data:
    #     csv_writer.writerow(row)
    # f_out.close()

    for key, value in counter.items():
        print(key, value, value * 0.9)

    for url in future_todo:
        print(url)

if __name__ == "__main__":
    main()