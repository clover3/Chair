import os
from typing import List, Iterable, Tuple

from arg.counter_arg.eval import load_problems
from arg.counter_arg.header import splits, ArguDataPoint
from arg.counter_arg.methods.bm25_predictor import get_bm25_module
from arg.counter_arg.methods.tool import sent_tokenize_newline, get_term_importance
from epath import job_man_dir
from galagos.interface import format_query_bm25, DocQuery
from galagos.parse import save_queries_to_file
from list_lib import lmap
from misc_lib import exist_or_mkdir


def main():
    for split in splits:
        queries: Iterable[DocQuery] = make_query(split)
        write_queries(split, list(queries))


def write_queries(split: str, queries: List[DocQuery]):
    root_dir_path = os.path.join(job_man_dir, "counter_arg_queries")
    exist_or_mkdir(root_dir_path)
    dir_path = os.path.join(root_dir_path, split)
    exist_or_mkdir(dir_path)
    query_per_file = 50
    file_idx = 0
    while file_idx * query_per_file < len(queries):
        save_path = os.path.join(dir_path, str(file_idx) + ".json")
        st = file_idx * query_per_file
        ed = st + query_per_file
        queries_to_save = queries[st:ed]
        save_queries_to_file(queries_to_save, save_path)
        file_idx += 1


def make_query(split) -> Iterable[DocQuery]:
    max_q_terms = 15
    k = 0.7
    problems: List[ArguDataPoint] = load_problems(split)
    # split is only used for total number of documents
    bm25 = get_bm25_module(split)

    def get_query_for_problem(problem: ArguDataPoint) -> List[str]:
        terms = list(get_term_score_for_problem(problem))
        _, last_term_score = terms[-1]

        q_terms = []
        for term, score in terms:
            norm_score = int(score / last_term_score)
            for _ in range(norm_score):
                q_terms.append(term)
        return q_terms

    def get_term_score_for_problem(problem: ArguDataPoint) -> Iterable[Tuple[str, float]]:
        text = problem.text1.text
        sents = sent_tokenize_newline(text)
        importance = get_term_importance(bm25, sents)
        for term, score in importance.most_common(max_q_terms):
            yield term, score

    def get_problem_id(p: ArguDataPoint) -> str:
        return p.text1.id.id

    problem_ids: List[str] = lmap(get_problem_id, problems)
    q_terms_list: List[List[str]] = lmap(get_query_for_problem, problems)
    for q_id, q_terms in zip(problem_ids, q_terms_list):
        yield format_query_bm25(q_id, q_terms, k)


if __name__ == "__main__":
    main()