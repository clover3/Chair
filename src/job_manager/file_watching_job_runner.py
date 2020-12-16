import os
import time
from typing import Callable, Any

from taskman_client.sync import JsonTiedDict


class FileWatchingJobRunner:
    def __init__(self,
                 watch_file_format_str: str,
                 info_path: str,
                 do_job_fn: Callable[[int], Any],
                 job_name):
        self.json_tied_dict = JsonTiedDict(info_path)
        self.watch_file_format_str: str = watch_file_format_str
        self.next_job_id = self.json_tied_dict.last_id() + 1
        self.job_name = job_name
        self.do_job = do_job_fn

    def start(self):
        terminate = False
        print("Waiting for job {}".format(self.next_job_id))
        while not terminate:
            watch_file_path = self.watch_file_format_str.format(self.next_job_id)
            if os.path.exists(watch_file_path):
                print("new job : ", watch_file_path)
                st = time.time()
                job_id = self.next_job_id
                self.do_job(job_id)
                self.json_tied_dict.set('last_task_id', job_id)
                self.next_job_id += 1
                ed = time.time()
                print("Finished {}. Elapsed={}".format(self.job_name, ed-st))
            else:
                time.sleep(10)

