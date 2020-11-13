import sys
from typing import List

from mturk.parse_util import HitResult, ColumnName, HITScheme, RepeatedEntries, parse_file


def main():
    num_doc_per_hit = 10
    file_path = sys.argv[1]
    url = "url"

    claim_column = ColumnName("claim")
    url_columns = list([ColumnName(url + "{}".format(i + 1)) for i in range(num_doc_per_hit)])
    inputs = []
    inputs.extend(url_columns)
    inputs.append(claim_column)

    prefixes = list(["{}".format(i+1) for i in range(num_doc_per_hit)])
    postfix_list = ["0", "1", "2"]
    answer_units = [RepeatedEntries("score", prefixes, postfix_list)]
    hit_scheme = HITScheme(inputs, answer_units)
    hr: List[HitResult] = parse_file(file_path, hit_scheme)

    for hit_result in hr:
        claim = hit_result.get_input(ColumnName("claim"))
        for i in range(num_doc_per_hit):
            name = ColumnName(url + "{}".format(i+1))
            doc_url = hit_result.get_input(name)
            radio_result = hit_result.get_repeated_entries_result("score", i)
            print([doc_url, claim, radio_result])


if __name__ == "__main__":
    main()