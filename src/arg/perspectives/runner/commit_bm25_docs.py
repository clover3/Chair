import time

from cache import load_from_pickle
from datastore.tool import commit_buffer_to_db_batch


def run():
    for i in range(0, 216):
        try:##
            print(i)
            save_name = "docs_BM25_100.jsonl_{}".format(i)
            begin = time.time()
            payload_saver = load_from_pickle(save_name)
            commit_buffer_to_db_batch(payload_saver.buffer)
            end = time.time()
            print(end-begin)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    run()