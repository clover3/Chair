import io
import sys

import msgpack

from misc_lib import TimeEstimator
from trainer_v2.chair_logging import c_log


def msgpack_test():
    c_log.info("msgpack_test Start")
    input_path = sys.argv[1]
    with open(input_path, "rb",) as f:
        c_log.info("Reading bytes")
        all_bytes = f.read()
        c_log.info("msgpack_test Start")
        c_log.info("Building bytes IO")
        stearm = io.BytesIO(all_bytes)
        c_log.info("Now unpacking")
        unpacker = msgpack.Unpacker(stearm, raw=False)
        itr = unpacker
        cnt = 0
        for j_item in itr:
            cnt += len(j_item)

    c_log.info("msgpack_test End")



def plain_text_read_write():
    input_path = sys.argv[1]
    f = open(input_path, "r")
    c_log.info("plain_text_read_write with split and keeping in memory")
    ticker = TimeEstimator(8841823)
    out_f = open("/tmp/out.txt", "w")
    array = []
    for line in f:
        tokens = line.split()
        array.append(tokens)
        out_f.write(" ".join(tokens) + "\n")
        ticker.tick()
    c_log.info("End")
    return NotImplemented


if __name__ == "__main__":
    msgpack_test()