import sys
from collections import defaultdict

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import group_by, get_first
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores


class NonExactError(Exception):
    pass


def main():
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    src_path = sys.argv[1]
    save_path = sys.argv[2]
    analyzer = EnglishAnalyzer()
    parser = QueryParser("contents", analyzer)
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(src_path)

    seen_d = {}
    def normalize(query_str) -> str:
        if query_str in seen_d:
            token_list = seen_d[query_str]
        else:
            es_query_str = parser.escape(query_str)
            query = parser.parse(es_query_str)

            struct_tokens = str(query).split()
            token_list = []
            for token in struct_tokens:
                field, term = token.split(":")
                token_list.append(term)
            seen_d[query_str] = token_list

        if len(token_list) == 1:
            norm_term = token_list[0]
            return norm_term
        else:
            raise NonExactError()

    q_grouped = defaultdict(list)
    for q_term, entries in value_mapping.items():
        try:
            norm_q_term = normalize(q_term)
            q_grouped[norm_q_term].extend(list(entries.items()))

        except NonExactError:
            pass

    new_table = {}
    print(f"Num query {len(value_mapping)} -> {len(q_grouped)}")
    for q_term, entries in q_grouped.items():
        out_entries = []
        for raw_d_term, score in entries:
            try:
                norm_d_term = normalize(raw_d_term)
                out_entries.append((norm_d_term, raw_d_term, score))
            except NonExactError:
                pass

        g = group_by(out_entries, get_first)
        per_q_d = {}
        for norm_d_term, entries in g.items():
            new_score = min([score for norm_d_term, raw_d_term, score in entries])
            per_q_d[norm_d_term] = new_score
        new_table[q_term] = per_q_d

    print("{} entries".format(sum(map(len, new_table.values()))))

    def triplet_iter():
        for q_term, entries in new_table.items():
            for d_term, score in entries.items():
                yield q_term, d_term, score

    save_tsv(triplet_iter(), save_path)



if __name__ == "__main__":
    main()