import csv
import os
from typing import List, Iterable, Dict, Tuple

from cpath import data_path
from data_generator.NLI.nli import get_modified_data_loader2, load_plain_text
from data_generator.NLI.nli_info import corpus_dir
from data_generator.shared_setting import BertNLI
from models.transformer import hyperparams

Indices = List[int]
ExplainInstance = Tuple[str, str, Indices, Indices]


def verify_explain_instance(insts: Iterable[ExplainInstance]):
    for p_text, h_text, p_indices, h_indices in insts:
        assert type(p_text) == str
        assert type(h_text) == str
        assert type(p_indices) == list
        assert type(h_indices) == list
        if p_indices:
            assert type(p_indices[0]) == int

def indice_str(indice):
    return ",".join(map(str, indice))


def save_to_tsv(insts: Iterable[ExplainInstance], save_path: str, data_id_prefix: str, data_id_base: int):
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab', lineterminator="\n")
    head_row = ["data_id", "premise", "hypothesis", "p_indices", "h_indices"]
    f_out.writerow(head_row)
    for idx, (p_text, h_text, p_indices, h_indices) in enumerate(insts):
        data_id = data_id_prefix + str(idx+data_id_base)
        row = [data_id, p_text, h_text, indice_str(p_indices), indice_str(h_indices)]
        f_out.writerow(row)


def reformat_conflict_dev(data: List[Dict]) -> List[ExplainInstance]:
    out_data = []
    for d in data:
        out_e = d['p'], d['h'], d['p_explain'], d['h_explain']
        assert d['y'] == 2
        out_data.append(out_e)
    return out_data


def save_dev_data():
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data_loader = get_modified_data_loader2(hp, nli_setting)
    _, explain_maybe_conflict_dev_raw = data_loader.get_dev_explain_0()
    _, explain_maybe_match_dev = data_loader.get_dev_explain("match")
    _, explain_maybe_mismatch_dev = data_loader.get_dev_explain("mismatch")

    explain_maybe_conflict_dev: List[ExplainInstance] = reformat_conflict_dev(explain_maybe_conflict_dev_raw)

    todo = {
        "match": explain_maybe_match_dev,
        "mismatch": explain_maybe_mismatch_dev,
        "conflict": explain_maybe_conflict_dev
    }

    prefix_d = {
        "match": "e",
        "mismatch": "n",
        "conflict": "c",
    }

    for name, data in todo.items():
        prefix = prefix_d[name]
        print(name, len(data))
        save_name = "{}_{}.tsv".format(prefix, "dev")
        save_path = os.path.join(data_path, "nli", "mnli_ex", save_name)
        save_to_tsv(data, save_path, prefix, 0)

    return NotImplemented


def read_gold_test(file_name):
    print(file_name)
    file_path = os.path.join(corpus_dir, file_name)
    reader2 = csv.reader(open(file_path, "r", encoding="utf-8"), delimiter=",")
    indice_list = []
    for row in reader2:
        id = int(row[0])
        p_indice, h_indice = row[1], row[2]
        p_indice = list([int(t) for t in p_indice.strip().split()])
        h_indice = list([int(t) for t in h_indice.strip().split()])
        indice_list.append((id, p_indice, h_indice))
    return indice_list


def save_test_data():
    prefix_d = {
        "match": "e",
        "mismatch": "n",
        "conflict": "c",
    }
    tags = ["match", "mismatch", "conflict"]
    # tags = ["mismatch"]

    def is_applicable(text, indices):
        if indices:
            return max(indices) < len(text.split())
        else:
            return True

    for tag in tags:
        texts = load_plain_text("{}_1000.csv".format(tag))
        for p, h in texts:
            assert h

        data_slice = texts[100:700]
        indices: List[Tuple[int, List, List]] = read_gold_test("gold_{}_100_700.csv".format(tag))
        prefix = prefix_d[tag]
        data = []
        for idx, (e1, e2) in enumerate(zip(data_slice, indices)):
            p_text, h_text = e1
            data_id, p_indices, h_indices = e2
            assert idx + 100 == data_id
            assert is_applicable(p_text, p_indices)
            assert is_applicable(h_text, h_indices)
            out_e = p_text, h_text, p_indices, h_indices
            data.append(out_e)

        save_name = "{}_{}.tsv".format(prefix, "test")
        save_path = os.path.join(data_path, "nli", "mnli_ex", save_name)
        save_to_tsv(data, save_path, prefix, 100)


def main():
    save_dev_data()


if __name__ == "__main__":
    main()