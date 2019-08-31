import csv



def load_stance_annot(path):
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)

    head = list(data[0])

    col_a_id = head.index("AssignmentId")
    col_idx_status = head.index("AssignmentStatus")
    col_statement_idx = head.index("Input.statement")
    col_e1 = head.index("Answer.E1")
    col_e2 = head.index("Answer.E2")
    col_r11 = head.index("Answer.r1.1.1")
    col_r12 = head.index("Answer.r1.1.2")
    col_r1_notfound = head.index("Answer.r1.1NotFound")
    col_r21 = head.index("Answer.r2.2.1")
    col_r22 = head.index("Answer.r2.2.2")
    col_r2_notfound = head.index("Answer.r2.2NotFound")

    NOT_FOUND = 0
    DIRECT_SUPPORT = 1
    INDIRECT_SUPPORT = 2

    parsed_data = []
    for entry in data[1:]:

        if entry[col_idx_status] in ["Rejected", "ManReject"]:
            continue

        statement = entry[col_statement_idx]

        r1 = NOT_FOUND
        if entry[col_r11] == "true":
            r1 = DIRECT_SUPPORT
        elif entry[col_r12] == "true":
            r1 = INDIRECT_SUPPORT
        else:
            r1 = NOT_FOUND

        r2 = NOT_FOUND
        if entry[col_r21] == "true":
            r2 = DIRECT_SUPPORT
        elif entry[col_r22] == "true":
            r2 = INDIRECT_SUPPORT
        else:
            r2 = NOT_FOUND

        evi1 = entry[col_e1]
        support_evidence = evi1.split(",")

        evi2 = entry[col_e2]
        dispute_evidence = evi2.split(",")

        d_entry = {}
        d_entry['statement'] = statement
        d_entry['support']  = r1
        d_entry['support_evidence'] = support_evidence
        d_entry['dispute'] = r2
        d_entry['dispute_evidence'] = dispute_evidence
        parsed_data.append(d_entry)
    return parsed_data


