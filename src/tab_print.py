def join_with_tab(l):
    return "\t".join([str(t) for t in l])


def tab_print(*args):
    return print(join_with_tab(args))


def print_table(rows):
    for row in rows:
        print(join_with_tab(row))


def tab_print_dict(dict):
    for k, v in dict.items():
        print("{}\t{}".format(k, v))


def save_table_as_tsv(rows, save_path):
    f = open(save_path, "w")
    for r in rows:
        f.write(join_with_tab(r) + "\n")
    f.close()


def save_table_as_csv(output_rows, save_path):
    import csv
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))

    for row in output_rows:
        csv_writer.writerow(map(str, row))


def concat_table_horizontally(table_list):
    if len(table_list) == 0:
        return []
    l = len(table_list[0])
    for t in table_list:
        assert len(t) == l

    new_table = []
    for row_i in range(l):
        new_row = []
        for j in range(len(table_list)):
            new_row.extend(table_list[j][row_i])
            new_row.append("")
        new_table.append(new_row)
    return new_table
