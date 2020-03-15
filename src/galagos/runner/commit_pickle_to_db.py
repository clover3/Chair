from cache import load_from_pickle
from datastore.tool import commit_buffer_to_db


def run():
    for i in range(0, 122):
        try:
            name = "{}.jsonl".format(i)
            payload_saver = load_from_pickle(name)
            payload_saver.commit_to_db()
            commit_buffer_to_db(payload_saver.buffer)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    run()