import os
import random
import itertools
import time

class MarkedTaskManager:
    def __init__(self, task_max, mark_dir, interval):
        self.task_max = task_max
        self.mark_dir = mark_dir
        if not os.path.exists(mark_dir):
            os.mkdir(mark_dir)
        self.interval = interval
        self.job_max = int(task_max / interval)+1
        # A job is composed of multiple task
        # Job[i] is responsible for task[h*i :h*(i+1)]

    def get_mark_path(self, job_id):
        return os.path.join(self.mark_dir, "{}.mark".format(job_id))

    def is_job_assigned(self, job_id):
        mark_path = self.get_mark_path(job_id)
        return os.path.exists(mark_path)

    def mark_job(self, job_id):
        f = open(self.get_mark_path(job_id), "w")
        f.write("mark")
        f.close()

    def pool_job(self):
        scan_begin = random.randint(0, self.job_max-1)

        scan_range = itertools.chain(range(scan_begin, self.job_max), range(scan_begin))
        for i in scan_range:
            if not self.is_job_assigned(i):
                self.mark_job(i)
                return i

        return None


class MTM:
    def __init__(self, job_max, mark_dir):
        self.mark_dir = mark_dir
        if not os.path.exists(mark_dir):
            os.mkdir(mark_dir)
        self.job_max = job_max

    def get_mark_path(self, job_id):
        return os.path.join(self.mark_dir, "{}.mark".format(job_id))

    def is_job_assigned(self, job_id):
        mark_path = self.get_mark_path(job_id)
        return os.path.exists(mark_path)

    def mark_job(self, job_id):
        f = open(self.get_mark_path(job_id), "w")
        f.write("mark")
        f.close()

    def pool_job(self):
        scan_begin = random.randint(0, self.job_max-1)

        scan_range = itertools.chain(range(scan_begin, self.job_max), range(scan_begin))
        for i in scan_range:
            if not self.is_job_assigned(i):
                self.mark_job(i)
                return i

        return None


class ReadyMarkTaskManager:
    def __init__(self, job_max, ready_sig, mark_dir):
        self.job_max = job_max
        self.ready_sig = ready_sig
        self.mark_dir = mark_dir
        if not os.path.exists(mark_dir):
            os.mkdir(mark_dir)

    def get_mark_path(self, job_id):
        return os.path.join(self.mark_dir, "{}.mark".format(job_id))

    def is_job_assigned(self, job_id):
        mark_path = self.get_mark_path(job_id)
        return os.path.exists(mark_path)

    def mark_job(self, job_id):
        f = open(self.get_mark_path(job_id), "w")
        f.write("mark")
        f.close()

    def is_job_ready(self, job_id):
        p = self.ready_sig.format(job_id)
        if os.path.exists(p):
            s = os.stat(p)
            return s.st_size > 0
        else:
            return False

    def pool_job_inner(self):
        scan_begin = random.randint(0, self.job_max-1)

        scan_range = itertools.chain(range(scan_begin, self.job_max), range(scan_begin))
        n_ready = 0
        for i in scan_range:
            if self.is_job_ready(i) :
                n_ready += 1
                if not self.is_job_assigned(i):
                    self.mark_job(i)
                    return i, n_ready

        return None, n_ready

    def pool_job(self):
        job_id, n_ready = self.pool_job_inner()

        while job_id is None and n_ready < self.job_max:
            time.sleep(60*2)
            job_id, n_ready = self.pool_job_inner()

        return job_id
