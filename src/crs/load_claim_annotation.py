import csv

def load_claim_annot(path):
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)

    head = list(data[0])

    col_a_id = head.index("AssignmentId")
    col_idx_status = head.index("AssignmentStatus")
    col_statements = list([head.index("Answer.statement{}".format(i)) for i in range(1,6)])
    col_url = head.index("Input.url")

    parsed_data = []
    for entry in data[1:]:

        if entry[col_idx_status] in ["Rejected", "ManReject"]:
            continue

        statements = []

        for i in range(5):
            idx = col_statements[i]
            statements.append(entry[idx])

        url = entry[col_url]

        d_entry = {}
        d_entry['statements'] = statements
        d_entry['url'] = url
        parsed_data.append(d_entry)
    return parsed_data


