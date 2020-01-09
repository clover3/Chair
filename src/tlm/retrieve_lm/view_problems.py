from tlm.retrieve_lm.tf_instance_writer import TFRecordMaker, get_candiset


def worker(tf_maker, job_id):
    cs = get_candiset(job_id)
    if cs is None:
        return

    info_list = []
    inst_list = []


    maker = tf_maker.generate_ir_tfrecord

    for j, e in enumerate(cs):
        for idx, problem in enumerate(maker(e)):
            print(problem)

    result = inst_list, info_list




if __name__ == "__main__":
    max_seq = 512
    tf_maker = TFRecordMaker(max_seq)
    worker(tf_maker , 0)
