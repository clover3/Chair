import sys
from collections import Counter
from typing import List, Iterable, Tuple

from list_lib import lmap
from misc_lib import group_by, get_second, get_third, average
from mturk.parse_util import HitResult, ColumnName, HITScheme, RepeatedEntries, parse_file


def load_parsed(file_path) -> List[Tuple[str, str, str]]:
    num_doc_per_hit = 10
    url = "url"
    claim_column = ColumnName("claim")
    url_columns = list([ColumnName(url + "{}".format(i + 1)) for i in range(num_doc_per_hit)])
    inputs = []
    inputs.extend(url_columns)
    inputs.append(claim_column)
    prefixes = list(["{}".format(i + 1) for i in range(num_doc_per_hit)])
    postfix_list = ["0", "1", "2"]
    answer_units = [RepeatedEntries("score", prefixes, postfix_list)]
    hit_scheme = HITScheme(inputs, answer_units)
    hr: List[HitResult] = parse_file(file_path, hit_scheme)
    out_rows = []
    for hit_result in hr:
        claim = hit_result.get_input(ColumnName("claim"))
        for i in range(num_doc_per_hit):
            name = ColumnName(url + "{}".format(i + 1))
            doc_url = hit_result.get_input(name)
            radio_result = hit_result.get_repeated_entries_result("score", i)
            row = (doc_url, claim, radio_result)
            out_rows.append(row)
    return out_rows


def main():
    file_path = sys.argv[1]
    out_rows = load_parsed(file_path)

    grouped = group_by(out_rows, get_second)
    score_rows = []
    for claim_text, rows in grouped.items():
        score_str_list: Iterable[str] = map(get_third, rows)
        score_list: Iterable[int] = map(int, score_str_list)
        cnt = Counter(score_list)
        row = (cnt[0], cnt[1], cnt[2])
        score_rows.append(row)

    for col_idx in [0, 1, 2]:
        avg_cnt = average(lmap(lambda row: row[col_idx], score_rows))
        print(col_idx, avg_cnt)


if __name__ == "__main__":
    main()