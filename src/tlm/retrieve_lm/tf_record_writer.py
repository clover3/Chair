import os
import pickle
import random

import cpath
from data_generator import tokenizer_wo_tf as tokenization
from job_manager.marked_task_manager import MarkedTaskManager
from tlm.two_seg_pretraining import write_instance_to_example_files, write_predict_instance


def filter_instances(data):
    inst_list, info_list = data

    n_inst_list, n_info_list = [], []
    for inst, info in zip(inst_list, info_list):
        if len(inst.tokens) > 512:
            continue
        n_info_list.append(info)
        n_inst_list.append(inst)

    n_drop = len(info_list) - len(n_info_list)
    if n_drop:
        print("Drop ", n_drop)
    return n_inst_list, n_info_list



def worker(job_id):
    max_seq = 512
    print("TF_record_writer")
    rng = random.Random(0)
    vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    p = os.path.join(cpath.data_path , "tlm", "instances_local", "inst_{}.pickle".format(job_id))
    if not os.path.exists(p):
        return
    output_path = os.path.join(cpath.data_path , "tlm", "tf_record_local", "tf_rand_{}.pickle".format(job_id))
    if os.path.exists(output_path):
        return
    inst_list, info_list = filter_instances(pickle.load(open(p, "rb")))

    rng.shuffle(inst_list)
    max_pred = 20
    print(inst_list[0])
    write_instance_to_example_files(inst_list, tokenizer, max_seq, max_pred, [output_path])


def worker_p(job_id):
    max_seq = 512
    vocab_file = os.path.join(cpath.data_path, "bert_voca.txt")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    p = os.path.join(cpath.data_path, "tlm", "instances", "inst_{}.pickle".format(job_id))
    if not os.path.exists(p):
        return
    output_path = os.path.join(cpath.data_path, "tlm", "tf_record_pred", "tf_{}.pickle".format(job_id))
    #if os.path.exists(output_path):
    #    return
    inst_list, info_list = filter_instances(pickle.load(open(p, "rb")))

    uid_list = []
    info_d = {}
    for inst, info in zip(inst_list, info_list):
        a,b,c = info.split("_")
        unique_id = int(a) * 1000*1000 + int(b) * 10 + int(c)
        uid_list.append(unique_id)
        info_d[unique_id] = info

    max_pred = 20
    data = zip(inst_list, uid_list)

    p = os.path.join(cpath.data_path, "tlm", "pred", "info_d_{}.pickle".format(job_id))
    pickle.dump(info_d, open(p, "wb"))
    write_predict_instance(data, tokenizer, max_seq, max_pred, [output_path])



def main():
    print("TF_record_writer")

    mark_path = os.path.join(cpath.data_path, "tlm", "tf_record_pred_mark")
    mtm = MarkedTaskManager(1000*1000, mark_path, 1000)
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker_p(job_id)
        job_id = mtm.pool_job()




if __name__ == "__main__":
    main()
