from cpath import at_output_dir
from dataset_specific.clueweb.query_loader import TrecQuery, load_queries
from galagos.interface import DocQuery
from galagos.parse import save_queries_to_file
from galagos.tokenize_util import clean_text_for_query
from list_lib import lmap

KEYWORD_QUERY = 1
DESC_QUERY = 2
all_years = range(2009, 2015)


def trec_query_to_galago_query(q: TrecQuery, query_type):
    if query_type == KEYWORD_QUERY:
        text = q.keyword_query
    elif query_type == DESC_QUERY:
        text = q.desc_query
    else:
        assert False

    text: str = clean_text_for_query(text)

    return DocQuery({
        'number': q.query_id,
        'text': text
    })


def work(years, query_type, save_path):
    queries = load_queries(years)

    def convert_query(q):
        return trec_query_to_galago_query(q, query_type)

    queries = lmap(convert_query, queries)
    save_queries_to_file(queries, save_path)


def main():
    work(range(2009, 2013), KEYWORD_QUERY, at_output_dir("clueweb", "keyword_09b_query.json"))
    work(range(2009, 2013), DESC_QUERY, at_output_dir("clueweb", "desc_09b_query.json"))
    work(range(2013, 2015), KEYWORD_QUERY, at_output_dir("clueweb", "keyword_12b_query.json"))
    work(range(2013, 2015), DESC_QUERY, at_output_dir("clueweb", "desc_12b_query.json"))


def debug_clean_text():
    queries = load_queries(all_years)

    def convert_query(q):
        return trec_query_to_galago_query(q, KEYWORD_QUERY)

    new_queries = lmap(convert_query, queries)

    for q_old, q_new in zip(queries, new_queries):
        if q_new['text'] != q_old.keyword_query:
            print("before:", q_old.keyword_query)
            print("after:", q_new['text'])


if __name__ == "__main__":
    debug_clean_text()