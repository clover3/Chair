import os
import pickle

import numpy as np

from cpath import output_path
from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from misc_lib import get_dir_files


def do():
    path = os.path.join(output_path, "clueweb12_agree")
    show(path)

def show(dir_path):
    topic = "abortion"
    tokenizer = get_tokenizer()
    for file_path in get_dir_files(dir_path):
        if topic not in file_path:
            continue
        file_name = os.path.basename(file_path)
        predictions = pickle.load(open(file_path, "rb"))
        for doc in predictions:
            show_doc = False
            for e in doc:
                sout, input_ids = e
                if sout[2] > 0.5 :
                    show_doc = True


            if show_doc:
                for e in doc:
                    sout, input_ids = e
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    pred = np.argmax(sout)
                    print(pred, pretty_tokens(tokens, True))
                print("------------")


if __name__ == "__main__":
    do()
