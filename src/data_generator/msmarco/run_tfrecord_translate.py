import pickle
import sys

from data_generator.msmarco.tfrecord_translator import translate


def main(file_path, out_path, st, ed):
    print(file_path)
    itr = translate(file_path, st, ed)
    data = list(itr)
    pickle.dump(data, open(out_path, "wb"))




if __name__ == "__main__":
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    st = int(sys.argv[3])
    ed = int(sys.argv[4])
    main(file_path, out_path, st, ed )