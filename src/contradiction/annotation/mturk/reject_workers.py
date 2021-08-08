import csv
from collections import Counter
from typing import Dict


def get_worker_list():
    workers = [
        "A2HW64P1UA7SDC",
        "A203PCDP2FZ9HS",
        "A26O22YLTVK3YC",
        "A1VFC06OXCJV02",
        "AI7UHEJHW0Q6N",
        "AE0YSGK7SPZ04",
        "A2WFRC1S0RSPXE",
        "A30NK1YTGRYUKQ",
        "A2UF2UO8AW1OSW",
        "A15IP94AWDQ2Y4",
        "A2OLAYNMLZHHRZ",
        "A1LSSHKTJHSPWL",
        "A1LPMC2QS9QKV5",
        "A1NOWFRT8M16XB",
        "A1736M4ZULQPEW",
        "A3QH35HZO2MIDB",
        "APG9EK6P41THO",
        "A27Z8TKP0CYS3Q",
        "A29F936QOI7EQ5",
        "ACPL0W3LOLJQ7",
        "A1WE7CII5F6VRR",
    ]
    return workers


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
    workers = get_worker_list()

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