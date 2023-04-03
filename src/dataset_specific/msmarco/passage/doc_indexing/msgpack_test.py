import io
import numpy as np
import h5py
import msgpack

from misc_lib import TimeEstimator
from trainer_v2.chair_logging import c_log


def enum_simulated_inv_index():
    f = open("posting_length.txt", "r")

    for line in f:
        term, n_posting = line.split("\t")
        posting = [(2828289, 123)] * int(n_posting)
        yield (term, posting)


def small_test():
    million = 1000000
    n_per_post = 1000
    item = [('doc', 3)] * n_per_post
    print(len(item))
    packer = msgpack.Packer()
    bytes_per_term = packer.pack(item)
    # for i in range(n_per_post):
    #     b = packer.pack(item)
    #     bytes_per_term += b
    print('bytes_per_term', len(bytes_per_term))
    all_bytes = b''
    for i in range(100):
        all_bytes += bytes_per_term

    print('all_bytes', len(all_bytes))

    stearm = io.BytesIO(all_bytes)
    unpacker = msgpack.Unpacker(stearm, raw=False)
    ticker = TimeEstimator(million, sample_size=10)
    cnt = 0
    for j_item in unpacker:
        cnt += len(j_item)
        ticker.tick()
        print("t")


def main():
    n_term = 4389837
    ticker = TimeEstimator(n_term)
    c_log.info("Writing postings with h5py")
    save_path = "/tmp/test_posting_out"
    # with open(save_path, "w") as f:
    with h5py.File(save_path, "w") as f:
        packer = msgpack.Packer()
        itr = enum_simulated_inv_index()
        for data in itr:
            term, postings = data
            doc_ids, counts = zip(*postings)
            doc_ids_i = [int(doc_id) for doc_id in doc_ids]
            l = len(doc_ids_i)
            arr = np.array([doc_ids, counts], np.int32)
            f.create_dataset(term, data=arr, dtype='i')
            # f.write(packer.pack(postings))
            ticker.tick()


    c_log.info("DONE")

if __name__ == "__main__":
    main()