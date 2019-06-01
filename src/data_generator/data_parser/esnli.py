import csv
from path import data_path
import os

scope_path = os.path.join(data_path, "nli", "e-snli")

def load_split(split_name):
    if split_name == "train":
        result = []
        for i in range(1,3):
            file_path = os.path.join(scope_path, "esnli_{}_{}.csv".format(split_name, i))
            result.extend(load_file(file_path))
        return result
        pass
    else:
        file_path = os.path.join(scope_path, "esnli_{}.csv".format(split_name))
        return load_file(file_path)


def load_file(path):
    f = open(path , "r")
    reader = csv.reader(f, delimiter=',')

    column_name = None
    data = []
    for idx, row in enumerate(reader):
        if idx == 0:
            column_name = row
        else:
            entry = {}
            for col_idx, col in enumerate(row):
                col_name = column_name[col_idx]
                entry[col_name] = col
            data.append(entry)
    return data


def load_gold(split_name):
    entries = parse_judgement(split_name)
    result = []
    for e in entries:
        ne = sorted(list(e['indice1'])), sorted(list(e['indice2']))
        result.append(ne)
    return result



def parse_judgement(split_name):
    r = load_split(split_name)

    for entry in r:
        all_indice = []
        for sent_id in [1,2]:
            indices = set()
            for i in range(1,4):
                key = "Sentence{}_Highlighted_{}".format(sent_id, i)
                if entry[key] != "{}":
                    indices.update([int(elem) for elem in entry[key].split(",")])
            all_indice.append(indices)

        entry['indice1'] = all_indice[0]
        entry['indice2'] = all_indice[1]

    return r

if __name__ == "__main__":

    r = parse_judgement("dev")
    for e in r[:2]:
        print(e)

    r = load_gold("dev")

    for e in r[:2]:
        print(e)