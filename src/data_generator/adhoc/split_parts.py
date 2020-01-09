import os
import pickle

import cpath


def split():
    q_id_list = [
        (301, 325),
        (326, 350),
        (351, 375),
        (376, 400),
        (401, 425),
        (426, 450),
        (601, 625),
        (626, 650),
        (651, 675),
        (676, 700),
    ]

    def find_range_idx(q_id):
        for i in range(10):
            st, ed = q_id_list[i]
            if st <= int(q_id) <= ed:
                return i

        assert False

    q_id_dict = dict()
    for q_id in range(301, 701):
        q_id_dict[q_id] = find_range_idx(q_id)

    payload_path = os.path.join(cpath.data_path, "robust_payload", "enc_payload_512.pickle")

    payload = pickle.load(open(payload_path, "rb"))

    payload_parts = list([list() for i in range(10)])
    for doc_id, q_id, runs in payload:
        idx = q_id_dict[q_id]
        payload_parts[idx].append((doc_id, q_id, runs))

    for i in range(10):
        payload_part_path = os.path.join(cpath.data_path, "robust_payload", "enc_payload_512_{}.pickle".format(i))
        pickle.dump(payload_parts, open(payload_part_path, "wb"))


if __name__ == '__main__':
    split()
