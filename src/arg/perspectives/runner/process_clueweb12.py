import os
import pickle
import time

from data_generator.common import get_tokenizer
from data_generator.job_runner import sydney_working_dir, JobRunner
from datastore.tool import PayloadSaver, commit_buffer_to_db
from galagos.tokenize_doc_and_save import tokenize_doc_and_save
from list_lib import lmap
from misc_lib import TimeEstimator


def get_dir_name_list():
    prefix = "/mnt/nfs/work3/youngwookim/data/clueweb12_text/1_00"
    f = open("/mnt/nfs/work3/youngwookim/data/clueweb_meta/1_00_warc_list.txt", "r")

    lines = lmap(lambda x: x.strip(), f)
    dir_names = lmap(os.path.basename, lines)
    dir_path_list = lmap(lambda x:os.path.join(prefix, x), dir_names)
    return dir_path_list



def work(warc_extracted_dir, url_to_doc_id, tokenize_fn):
    f = open(os.path.join(warc_extracted_dir, "idx_to_url"), "rb")
    idx_to_url = pickle.load(f)
    payload_saver = PayloadSaver()
    ticker = TimeEstimator(len(idx_to_url))
    for idx in idx_to_url:
        url = idx_to_url[idx]
        url = url.replace(",","")
        ##
        doc_id = url_to_doc_id[url]
        txt_path = os.path.join(warc_extracted_dir, "{}.txt".format(idx))
        ticker.tick()
        if os.path.exists(txt_path):
            text = open(txt_path).read()
            tokenize_doc_and_save(payload_saver, doc_id, text, tokenize_fn)
            # open filen idx_to_url
    return payload_saver


def work_multithread(warc_extracted_dir, url_to_doc_id, tokenize_fn):
    f = open(os.path.join(warc_extracted_dir, "idx_to_url"), "rb")
    idx_to_url = pickle.load(f)

    ticker = TimeEstimator(len(idx_to_url))
    todo_list = []
    work_size = 0
    max_job_size = 1000 * 1000 * 2 # 100MB
    num_thread = 10
    last_payload = PayloadSaver()
    def launch_task():
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        def list_fn(todo_list) -> PayloadSaver:
            payload_saver = PayloadSaver()
            for doc_id, text in todo_list:
                tokenize_doc_and_save(payload_saver, doc_id, text, tokenize_fn)
            return payload_saver

        from pathos.multiprocessing import ProcessingPool as Pool
        p = Pool(num_thread, daemon=True)

        split_n = int(len(todo_list) / num_thread) + 1
        args = chunks(todo_list, split_n)
        result_handle = p.amap(list_fn, args)
        return result_handle

    def launch_and_commit():
        print("launching task")
        results = launch_task()

        nonlocal last_payload
        print("commit buffer to db")
        commit_buffer_to_db(last_payload.buffer)

        print("wait tasks to be done")
        while not results.ready():
            time.sleep(5)

        print("combine payload")
        last_payload = combine_payload(results.get())

    def combine_payload(result_list_list):
        comb_payload_saver = PayloadSaver()
        for ps in result_list_list:
            comb_payload_saver.buffer.extend(ps.buffer)
        return comb_payload_saver

    print("Reading docs")
    for idx in idx_to_url:
        url = idx_to_url[idx]
        url = url.replace(",", "")
        ##
        doc_id = url_to_doc_id[url]
        txt_path = os.path.join(warc_extracted_dir, "{}.txt".format(idx))
        ticker.tick()
        try:
            text = open(txt_path).read()
            todo_list.append((doc_id, text))
            work_size += len(text)
        except FileNotFoundError:
            pass

        if work_size > max_job_size:
            launch_and_commit()
            print("Reading docs")
            work_size = 0
            todo_list = []

    commit_buffer_to_db(last_payload.buffer)


class JsonlWorker:
    def __init__(self, out_path_not_used):
        self.jsonl_path_format = "/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/doc_jsonl/{}.jsonl"
        self.dir_path_list = get_dir_name_list()
        self.tokenize_fn = get_tokenizer().tokenize

    def work(self, job_id):
        dir_path = self.dir_path_list[job_id]
        url_to_doc_id = pickle.load(open(os.path.join(dir_path, 'url_to_doc_id'), "rb"))
        work_multithread(dir_path, url_to_doc_id, self.tokenize_fn)
        # save_name = "clue_disk1_00_{}".format(job_id)
        # save_to_pickle(payload_saver, save_name)


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 1444, "clue_disk_1", JsonlWorker)
    runner.start()

