import sys
from collections import OrderedDict
from typing import Callable

from trainer_v2.per_project.cip.tfrecord_gen import LabeledInstance, encode_together, build_encoded, \
    SelectOneToOne, ItemSelector, SelectUpToK, SelectAll, encode_separate, encode_three, encode_four


def seq300wrap(
        name: str,
        selector: ItemSelector,
        encode_fn_inner: Callable[[int, LabeledInstance], OrderedDict]):

    if sys.argv[1] == name:
        pass
    else:
        return
    seq_length = 300
    print("Dataset name={}".format(name))

    def encode_fn(e: LabeledInstance) -> OrderedDict:
        return encode_fn_inner(seq_length, e)

    build_encoded(name, selector, encode_fn)


def main():
    # seq300wrap("cip1", SelectOneToOne(), encode_together)
    # seq300wrap("cip2", SelectUpToK(10), encode_together)
    # seq300wrap("cip1_eval", SelectAll(), encode_together)
    # seq300wrap("cip3", SelectOneToOne(), encode_separate)
    # seq300wrap("cip3_eval", SelectAll(), encode_separate)

    # seq300wrap("cip4", SelectOneToOne(), encode_three)
    # seq300wrap("cip4_eval", SelectAll(), encode_three)
    # seq300wrap("cip_d5", SelectOneToOne(), encode_four)
    seq300wrap("cip_d5_eval", SelectAll(), encode_four)

#
if __name__ == "__main__":
    main()
