import cpath
from cache import *
from galagos.parse import clean_query, get_query_entry, save_queries_to_file


def main():
    print("Start")
    spr = StreamPickleReader("robust_candi_query_")
    query_per_task = 1000 * 10
    out_idx = 0
    while spr.has_next():
        queries = []
        for i in range(query_per_task):
            if not spr.has_next():
                break
            q_id, query = spr.get_item()
            query = clean_query(query)
            queries.append(get_query_entry(q_id, query))

        out_path = os.path.join(cpath.output_path, "query", "g_query_{}.json".format(out_idx))
        save_queries_to_file(queries, out_path)
        out_idx += 1


if __name__ == "__main__":
    main()
