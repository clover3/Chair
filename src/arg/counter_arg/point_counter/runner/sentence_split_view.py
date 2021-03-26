from typing import List, Tuple

from nltk.tokenize import sent_tokenize

from arg.counter_arg.header import Passage
from arg.counter_arg.point_counter.prepare import load_data, load_data_from_pickle
from cache import save_to_pickle
from list_lib import lmap, lflatten, lfilter, lfilter_not
from misc_lib import Averager


def save_to_cache():
    for split in ["training", "validation"]:
        train_data: List[Tuple[Passage, int]] = load_data(split)
        save_name = "argu_pointwise_{}".format(split)
        save_to_pickle(train_data, save_name)


def main():
    train_data = load_data_from_pickle("training")
    averager = Averager()

    for text, label in train_data[:200]:
        print(label)
        raw_text = text.text
        text_list: List[str] = raw_text.split("\n\n")

        def is_empty_line(l):
            return l.strip()

        text_list = lfilter(is_empty_line, text_list)
        sentence_list = lflatten(lmap(sent_tokenize, text_list))

        def is_reference(l):
            if len(l) < 3 :
                return False
            if l[0] == "[" and l[1] == "i":
                return True
            if l[0] == "[" and l[2] == "]":
                return True
            if "http://" in l:
                return True
            return False

        sentence_list = lfilter_not(is_reference, sentence_list)
        averager.append(len(sentence_list))
    print(averager.get_average())





if __name__ == "__main__":
    main()