import os
import pickle
from collections import Counter

import numpy as np

from cpath import output_path
from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import pretty_tokens


def do():

    tokenizer = get_tokenizer()
    fn = "ClueWeb12-Disk1_01.idx_abortion.txt"
    fn = "ClueWeb12-Disk4_17.idx_gun_control.txt"
    obj = pickle.load(open(os.path.join(output_path, fn), "rb"))
    print(len(obj))

    content = {}
    cnt = 0
    for idx, doc in enumerate(obj):
        if not doc:
            continue

        show = []
        pred = Counter()
        for sent_info in doc:
            logits = sent_info[0]
            input_ids = sent_info[3]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            y = np.argmax(logits)
            pred[y] += 1
            show.append((tokens, y))

        content[idx] = show

        total = sum(pred.values())
        if pred[0] > total * 0.9:
            cnt += 1
            print("{} : {}/{}".format(idx, pred[0], total))

            for tokens, y in show:
                print(y, pretty_tokens(tokens, True))
    print(cnt)


if __name__ == "__main__":
    do()