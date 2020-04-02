from sqlalchemy.exc import IntegrityError

from cache import load_from_pickle
from datastore.tool import commit_buffer_to_db_batch


def run():
    for i in range(0, 128):
        try:
            print(i)
            name = "docs_10.jsonl_{}".format(i)
            payload_saver = load_from_pickle(name)
            #payload_saver.commit_to_db()
            commit_buffer_to_db_batch(payload_saver.buffer)
        except FileNotFoundError as e:
            print(e)
        except IntegrityError as e:
            print(e)


if __name__ == "__main__":
    run()