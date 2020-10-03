import sys

from arg.qck.dynamic_kdp.tfrecord_maker import TFRecordMaker

if __name__ == "__main__":
    worker = TFRecordMaker()
    job_id = int(sys.argv[1])
    worker.make_tfrecord(job_id)