import argparse
import os
import signal
import sys
import time

import psutil

from misc_lib import exist_or_mkdir, tprint
from taskman_client.sync import JsonTiedDict
from taskman_client.task_proxy import get_task_manager_proxy, get_local_machine_name

sh_path = os.path.join("task", "task_sh")
mark_dir = os.path.join("task", "task_mark")
log_path = os.path.join("task", "task_log")

exist_or_mkdir(sh_path)
exist_or_mkdir(mark_dir)
exist_or_mkdir(log_path)

info_path = os.path.join("task", "info.json")
task_info = JsonTiedDict(info_path)

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument("--max_job", default=10)



def preexec_function():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ActiveProcList:
    def __init__(self):
        self.proc_handles = []

    def add(self, proc):
        self.proc_handles.append(proc)

    def update_alive(self):
        tprint("check_active_tasks")
        task_to_mark_complete = []
        for task_process in self.proc_handles:
            try:
                status = task_process.status()
            except psutil.NoSuchProcess as e:
                status = "dead"

            if status == "running":
                pass
            elif status == "dead":
                task_to_mark_complete.append(task_process)

        for proc in task_to_mark_complete:
            self.proc_handles.remove(proc)
        return len(self.proc_handles)


def get_sh_path_for_job_id(job_id):
    return os.path.join(sh_path, "{}.sh".format(job_id))


def get_log_path(job_id):
    return os.path.join(log_path, "{}.log".format(job_id))


def get_last_mark():
    init_id = max(task_info.last_executed_task_id, 0)
    id_idx = init_id
    while os.path.exists(os.path.join(mark_dir, str(id_idx))):
        id_idx += 1

    task_info.set('last_executed_task_id', id_idx-1)
    return id_idx - 1


def check_job_mark(job_id):
    return os.path.exists(os.path.join(mark_dir, str(job_id)))


def get_new_job_id():
    init_id = task_info.last_executed_task_id
    id_idx = init_id
    while os.path.exists(get_sh_path_for_job_id(id_idx)):
        id_idx += 1

    return id_idx


def get_next_sh_path():
    return get_sh_path_for_job_id(get_new_job_id())


def get_next_sh_path_and_job_id():
    job_id = get_new_job_id()
    return get_sh_path_for_job_id(job_id), job_id


max_task = 30


def check_wait_tasks(active_proc_list):
    num_tas = active_proc_list.update_alive()
    tprint("Number of active task : ", num_tas)

    while num_tas > max_task:
        tprint("Waiting for tasks to be done")
        time.sleep(60)
        num_tas = active_proc_list.update_alive()


def loop(max_job):
    last_mask = get_last_mark()
    tprint("Last mark : ", last_mask)
    task_manager_proxy = get_task_manager_proxy()

    machine_name = get_local_machine_name()

    def is_machine_busy():
        active_jobs = task_manager_proxy.get_num_active_jobs(machine_name)
        pending_jobs = task_manager_proxy.get_num_pending_jobs(machine_name)
        tprint("{} active {} pending".format(active_jobs, pending_jobs))
        return active_jobs + pending_jobs > max_job

    no_job_time = 0
    while no_job_time < 1200:
        # check if there is additional job to run
        job_id = last_mask + 1
        next_sh_path = get_sh_path_for_job_id(job_id)
        if os.path.exists(next_sh_path):
            last_scheduled_job_id = get_new_job_id() - 1
            remaining_jobs = last_scheduled_job_id - job_id
            while is_machine_busy():
                tprint("Sleeping for jobs to be done. Remaining jobs : {}".format(remaining_jobs))
                time.sleep(10)
            execute(job_id)
            no_job_time = 0
            task_info.set("last_executed_task_id", job_id)
            mark(job_id)
            last_mask += 1
            time.sleep(5)
        else:
            no_job_time += 10
            tprint("no job time: ", no_job_time)
            time.sleep(10)

    tprint("Terminating")


def mark(job_id):
    open(os.path.join(mark_dir, str(job_id)), "w").close()


def execute(job_id):
    out = open(get_log_path(job_id), "w")
    p = psutil.Popen(["/bin/bash", get_sh_path_for_job_id(job_id)],
                     stdout=out,
                     stderr=out,
                     preexec_fn=preexec_function
                     )
    tprint("Executed job {} .  pid={}".format(job_id, p.pid))
    return p


if __name__ == "__main__":
    try:
        args = arg_parser.parse_args(sys.argv[1:])
        max_job = int(args.max_job)
        loop(max_job)
    except KeyboardInterrupt:
        tprint("Terminating with Keyboard interrupt")