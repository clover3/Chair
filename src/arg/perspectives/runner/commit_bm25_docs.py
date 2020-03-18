from cache import load_from_pickle
from datastore.tool import commit_buffer_to_db


def run():
    for i in range(6, 50):
        try:##
            print(i)
            save_name = "bm25_res_docs.jsonl_{}".format(i)
            payload_saver = load_from_pickle(save_name)
            commit_buffer_to_db(payload_saver.buffer)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    run()