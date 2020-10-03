import os
import pickle
from collections import OrderedDict
from concurrent.futures.process import ProcessPoolExecutor
from typing import List

import tensorflow as tf

from arg.perspectives.runner_qck.qck_gen_dynamic_kdp_val import get_qck_gen_dynamic_kdp
from arg.qck.dynamic_kdp.qck_generator import QCKGenDynamicKDP
from misc_lib import DataIDManager, tprint, ceil_divide

tf = tf.compat.v1



def enc(features_list):
    output = []
    for features in features_list:
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        serialized = tf_example.SerializeToString()
        output.append(serialized)
    return output


def serialize(features_list):
    num_worker = 4
    output = []
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        future_list = []
        job_per_worker = ceil_divide(len(features_list), num_worker)
        for idx in range(num_worker):
            st = idx * job_per_worker
            ed = (idx+1) * job_per_worker
            future = executor.submit(enc, features_list[st:ed])
            future_list.append(future)

        for future in future_list:
            sub_outputs = future.result()
            output.extend(sub_outputs)
    return output


class RecordWriterWrap:
    def __init__(self, outfile):
        self.writer = tf.python_io.TFRecordWriter(outfile)
        self.total_written = 0

    def write_feature_list(self, features_list):
        s_list = serialize(features_list)

        for s in s_list:
            self.writer.write(s)
        self.total_written += 1

    def close(self):
        self.writer.close()


def write_records_w_encode_fn_mt(output_path,
                              encode,
                              records,
                              ):
    writer = RecordWriterWrap(output_path)
    tprint("Making records")
    features_list: List[OrderedDict] = list(map(encode, records))
    tprint("Total of {} records".format(len(features_list)))
    writer.write_feature_list(features_list)
    writer.close()

    tprint("Done writing")



if __name__ == "__main__":
    data_id_manager = DataIDManager(0, 1000 * 1000)
    job_id = 4
    request_dir = os.environ["request_dir"]
    save_path = os.path.join(request_dir, str(job_id))
    kdp_list = pickle.load(open(save_path, "rb"))
    kdp_list = kdp_list[:2]
    qck_generator: QCKGenDynamicKDP = get_qck_gen_dynamic_kdp()
    tf_record_dir = os.environ["tf_record_dir"]

    print("{} kdp".format(len(kdp_list)))
    insts = qck_generator.generate(kdp_list, data_id_manager)
    record_save_path = os.path.join(tf_record_dir, str(job_id) + ".test")
    write_records_w_encode_fn_mt(record_save_path, qck_generator.encode_fn, insts)
