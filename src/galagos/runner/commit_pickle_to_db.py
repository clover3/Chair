import pickle
import sys

from datastore.tool import commit_buffer_to_db_batch


def run():
    prefix = sys.argv[1]
    num_items = int(sys.argv[2])
    for i in range(0, num_items):
        try:
            path = prefix + "_{}.pickle".format(i)
            payload_saver = pickle.load(open(path, "rb"))
            commit_buffer_to_db_batch(payload_saver.buffer)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    run()