from cache import load_from_pickle
from datastore.tool import commit_buffer_to_db_batch


def run():
    for i in range(0, 2316):
        try:
            print(i)
            name = "ca_docs.jsonl_{}".format(i)
            payload_saver = load_from_pickle(name)
            commit_buffer_to_db_batch(payload_saver.buffer)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    run()