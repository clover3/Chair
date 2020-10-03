import os
import pickle

from arg.perspectives.runner_qck.qck_gen_dynamic_kdp_val import get_qck_gen_dynamic_kdp
from arg.qck.dynamic_kdp.qck_generator import QCKGenDynamicKDP
from cpath import output_path
from misc_lib import DataIDManager
from taskman_client.file_watching_job_runner import FileWatchingJobRunner
from taskman_client.sync import JsonTiedDict
from taskman_client.task_executer import get_next_sh_path_and_job_id
from tf_util.record_writer_wrap import write_records_w_encode_fn


def run_job(sh_format_path, arg_map):
    content = open(sh_format_path, "r").read()
    for key, arg_val in arg_map.items():
        content = content.replace(key, arg_val)

    sh_path, job_id = get_next_sh_path_and_job_id()
    content = content.replace("--job_id=-1", "--job_id={}".format(job_id))
    f = open(sh_path, "w")
    f.write(content)
    f.close()


def add_estimator_job(job_id):
    cppnc_predict_sh = "cppnc_auto.sh"

    run_job(cppnc_predict_sh, {
        '$1': str(job_id),
    })


class TFRecordMaker:
    def __init__(self):
        self.request_dir = os.environ["request_dir"]
        self.tf_record_dir = os.environ["tf_record_dir"]
        info_path = os.path.join(self.request_dir, "info.json")
        self.json_tied_dict = JsonTiedDict(info_path)
        self.next_job_id = self.json_tied_dict.last_id() + 1
        self.qck_generator: QCKGenDynamicKDP = get_qck_gen_dynamic_kdp()
        self.save_dir = os.path.join(output_path, "cppnc_auto")

        score_save_path_format = os.path.join(self.request_dir, "{}")
        self.job_runner = FileWatchingJobRunner(score_save_path_format,
                                                info_path,
                                                self.make_tfrecord,
                              "tfrecord maker")

        print("")
        print("  [ TFRecordMaker ]")
        print()

    def file_watch_daemon(self):
        self.job_runner.start()
        print("TFRecordMaker thread()")

    def make_tfrecord(self, job_id: int):
        save_path = os.path.join(self.request_dir, str(job_id))
        kdp_list = pickle.load(open(save_path, "rb"))
        data_id_manager = DataIDManager(0, 1000 * 1000)
        print("{} kdp".format(len(kdp_list)))
        insts = self.qck_generator.generate(kdp_list, data_id_manager)
        record_save_path = os.path.join(self.tf_record_dir, str(job_id))
        write_records_w_encode_fn(record_save_path, self.qck_generator.encode_fn, insts)
        # Save for backup
        info_save_path = os.path.join(self.tf_record_dir, "{}.info".format(job_id))
        pickle.dump(data_id_manager.id_to_info, open(info_save_path, "wb"))
        # launch estimator
        add_estimator_job(job_id)
        # calculate score for each kdp


if __name__ == "__main__":
    worker = TFRecordMaker()
    worker.file_watch_daemon()