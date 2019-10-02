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



def load_stance_verify_annot(path):
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)

    head = list(data[0])

    col_a_id = head.index("AssignmentId")
    col_idx_status = head.index("AssignmentStatus")
    ans_cols = []
    for i_pre in range(5):
        i = i_pre + 1
        ans_cols.append({
            'statement':head.index("Input.statement{}".format(i)),
            'link': head.index("Input.link{}".format(i)),
            '1.NoSupport':head.index("Answer.r{}.1.NoSupport".format(i)),
            '1.NotSure':head.index("Answer.r{}.1.NotSure".format(i)),
            '1.Support':head.index("Answer.r{}.1.Support".format(i)),
            '2.NoDispute':head.index("Answer.r{}.2.NoDispute".format(i)),
            '2.NotSure':head.index("Answer.r{}.2.NotSure".format(i)),
            '2.Dispute':head.index("Answer.r{}.2.Dispute".format(i)),
        })

    NOT_FOUND = 0
    YES = 1
    NOT_SURE = 2
    parsed_data = []
    for entry in data[1:]:
        if entry[col_idx_status] in ["Rejected", "ManReject"]:
            continue
        for i in range(5):
            cols = ans_cols[i]
            statement = entry[cols['statement']]
            link = entry[cols['link']]
            r1 = NOT_FOUND
            if entry[cols['1.Support']] == "true":
                r1 = YES
            elif entry[cols['1.NotSure']] == "true":
                r1 = NOT_SURE
            else:
                r1 = NOT_FOUND

            r2 = NOT_FOUND
            if entry[cols['2.Dispute']] == "true":
                r2 = YES
            elif entry[cols['2.NotSure']] == "true":

                r2 = NOT_SURE
            else:
                r2 = NOT_FOUND

            d_entry = {}
            d_entry['statement'] = statement
            d_entry['link'] = link
            d_entry['support']  = r1
            d_entry['dispute'] = r2
            parsed_data.append(d_entry)
    return parsed_data


