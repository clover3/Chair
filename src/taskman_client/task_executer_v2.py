import os
import signal
import time
from typing import NamedTuple

import psutil

from misc_lib import exist_or_mkdir, tprint
from taskman_client.sync import JsonTiedDict
from taskman_client.task_proxy import get_local_machine_name, get_task_manager_proxy


def preexec_function():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


STATE_NOT_STARTED = -1
STATE_PENDING = 0
STATE_STARTED = 1
STATE_RUNNING = 1
STATE_TERMINATED = 3


class TaskInfo(NamedTuple):
    job_id: int
    pid: int
    state: int



class TaskExecuter:
    def __init__(self, root_dir, max_job=10):
        self.root_dir = root_dir

        self.sh_path = os.path.join(root_dir, "task_sh")
        self.mark_dir = os.path.join(root_dir, "mark_dir")
        self.log_path = os.path.join(root_dir, "log_path")

        exist_or_mkdir(self.sh_path)
        exist_or_mkdir(self.mark_dir)
        exist_or_mkdir(self.log_path)

        self.info_path = os.path.join(root_dir, "info.json")
        self.task_info = JsonTiedDict(self.info_path)
        self.task_manager_proxy = get_task_manager_proxy()
        self.machine_name = get_local_machine_name()
        self.max_job = max_job

    def get_sh_path_for_job_id(self, job_id):
        return os.path.join(self.sh_path, "{}.sh".format(job_id))

    def get_log_path(self, job_id):
        return os.path.join(self.log_path, "{}.log".format(job_id))

    def ask_if_machine_is_busy(self):
        active_jobs = self.task_manager_proxy.get_num_active_jobs(self.machine_name)
        pending_jobs = self.task_manager_proxy.get_num_pending_jobs(self.machine_name)
        tprint("{} active {} pending".format(active_jobs, pending_jobs))
        return active_jobs + pending_jobs > self.max_job

    def get_last_mark(self):
        init_id = max(self.task_info.get('last_executed_task_id'), 0)
        id_idx = init_id
        while os.path.exists(os.path.join(self.mark_dir, str(id_idx))):
            id_idx += 1

        self.task_info.set('last_executed_task_id', id_idx - 1)
        return id_idx - 1

    def mark(self, job_id):
        open(os.path.join(self.mark_dir, str(job_id)), "w").close()

    def get_new_job_id(self):
        init_id = self.task_info.get('last_executed_task_id')
        id_idx = init_id
        while os.path.exists(self.get_sh_path_for_job_id(id_idx)):
            id_idx += 1
        return id_idx

    def execute(self, job_id):
        out = open(self.get_log_path(job_id), "w")
        p = psutil.Popen(["/bin/bash", self.get_sh_path_for_job_id(job_id)],
                         stdout=out,
                         stderr=out,
                         preexec_fn=preexec_function
                         )
        tprint("Executed job {} .  pid={}".format(job_id, p.pid))
        return p

    def start(self):
        no_job_time = 0
        last_mask = self.get_last_mark()
        tprint("Last mark : ", last_mask)
        while no_job_time < 1200:
            # check if there is additional job to run
            job_id = last_mask + 1
            next_sh_path = self.get_sh_path_for_job_id(job_id)
            if os.path.exists(next_sh_path):
                last_scheduled_job_id = self.get_new_job_id() - 1
                remaining_jobs = last_scheduled_job_id - job_id
                while self.ask_if_machine_is_busy():
                    tprint("Sleeping for jobs to be done. Remaining jobs : {}".format(remaining_jobs))
                    time.sleep(10)
                self.execute(job_id)
                no_job_time = 0
                self.task_info.set("last_executed_task_id", job_id)
                self.mark(job_id)
                last_mask += 1
                time.sleep(5)
            else:
                no_job_time += 10
                tprint("no job time: ", no_job_time)
                time.sleep(10)

        tprint("Terminating")

