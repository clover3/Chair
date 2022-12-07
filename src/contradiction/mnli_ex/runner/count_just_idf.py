from collections import defaultdict

from adhoc.clueweb12_B13_termstat import ClueIDF
from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex
from dataset_specific.mnli.mnli_reader import MNLIReader
from list_lib import lmap
from misc_lib import Averager
from tab_print import print_table


def main():
    idf_module = ClueIDF()
    def tokenize(text):
        tokens = text.lower().split()
        return [t for t in tokens if t[-1] not in ".?!\"\'"]

    reader = MNLIReader()
    averagers = defaultdict(Averager)

    def update_idfs(key, tokens):
        for s in lmap(idf_module.get_weight, tokens):
            averagers[key].append(s)

    for idx, item in enumerate(reader.load_split("train")):
        if idx > 10000:
            break
        try:
            p_tokens = tokenize(item.premise)
            h_tokens = tokenize(item.hypothesis)
            p_only = [t for t in p_tokens if t not in h_tokens]
            h_only = [h for h in h_tokens if h not in p_tokens]
            update_idfs('p_only', p_only)
            update_idfs('h_only', h_only)
            update_idfs('p_tokens', p_tokens)
            update_idfs('h_tokens', h_tokens)
            label = item.label
            update_idfs('p_only_{}'.format(label), p_only)
            update_idfs('h_only_{}'.format(label), h_only)
            update_idfs('p_tokens_{}'.format(label), p_tokens)
            update_idfs('h_tokens_{}'.format(label), h_tokens)

        except UnicodeDecodeError:
            pass

    table = []
    for key, avger in averagers.items():
        table.append([key, avger.get_average()])

    table.sort(key=lambda x: x[1], reverse=True)
    print_table(table)


def do_for_nli_ex():
    idf_module = ClueIDF()
    averagers = defaultdict(Averager)

    def update_idfs(key, tokens):
        for s in lmap(idf_module.get_weight, tokens):
            averagers[key].append(s)

    split = "test"
    target_tag = "mismatch"
    problems = load_mnli_ex(split, target_tag)

    for p in problems:
        p_tokens = p.premise.lower().split()
        h_tokens = p.hypothesis.lower().split()
        p_only = [t for t in p_tokens if t not in h_tokens]
        h_only = [h for h in h_tokens if h not in p_tokens]

        mismatch_tokens = [h_tokens[i] for i in p.h_indices]

        update_idfs('p_tokens', p_tokens)
        update_idfs('h_tokens', h_tokens)
        update_idfs('h_only', h_only)
        update_idfs('mismatch_tokens', mismatch_tokens)

    table = []
    for key, avger in averagers.items():
        table.append([key, avger.get_average()])

    table.sort(key=lambda x: x[1], reverse=True)
    print_table(table)


if __name__ == "__main__":
    do_for_nli_ex()