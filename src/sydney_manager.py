import os
import random
import itertools

class MarkedTaskManager:
    def __init__(self, task_max, mark_dir, interval):
        self.task_max = task_max
        self.mark_dir = mark_dir
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
