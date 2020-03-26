import os
import time
import uuid

from taskman_client.RESTProxy import RESTProxy


class TaskManagerProxy(RESTProxy):
    def __init__(self, host, port):
        super(TaskManagerProxy, self).__init__(host, port)

    def task_update(self, run_name, uuid, tpu_name, machine, update_type, msg, job_id=None):
        data = {
            'run_name': run_name,
            'uuid': uuid,
            'tpu_name': tpu_name,
            'machine': machine,
            'update_type': update_type,
            'msg': msg
        }
        if job_id is not None:
            data['job_id'] = job_id
        return self.post("/task/update", data)

    def task_pending(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "PENDING"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_start(self, run_name, uuid_var, tpu_name, machine, msg, job_id):
        update_type = "START"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg, job_id)

    def task_complete(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "SUCCESSFUL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_interrupted(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "ABNORMAL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def report_number(self, name, value, condition):
        data = {
            'name': name,
            "number": value,
            "condition": condition,
        }
        return self.post("/experiment/update", data)

    def get_tpu(self, tpu_condition=None):
        data = {
            "v": tpu_condition
        }
        return self.post("/task/get_tpu", data)

    def get_num_active_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_active_jobs", data)
        return r['num_active_jobs']

    def get_num_pending_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_pending_jobs", data)
        return r['num_pending_jobs']

class TaskProxy:
    def __init__(self, host, port, machine, tpu_name=None, uuid_var=None):
        self.proxy = TaskManagerProxy(host, port)
        self.tpu_name = tpu_name
        self.machine = machine

        if uuid_var is None:
            self.uuid_var = str(uuid.uuid1())
        else:
            self.uuid_var = uuid_var

    def task_pending(self, run_name, msg=None):
        if run_name == "dontreport":
            return
        return self.proxy.task_pending(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_start(self, run_name, msg=None, job_id=None):
        if run_name == "dontreport":
            return
        return self.proxy.task_start(run_name, self.uuid_var, self.tpu_name, self.machine, msg, job_id)

    def task_complete(self, run_name, msg=None):
        if run_name == "dontreport":
            return
        return self.proxy.task_complete(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_interrupted(self, run_name, msg=None):
        if run_name == "dontreport":
            return
        return self.proxy.task_interrupted(run_name, self.uuid_var, self.tpu_name, self.machine, msg)


def get_local_machine_name():
    if os.name == "nt":
        return os.environ["COMPUTERNAME"]
    else:
        return os.uname()[1]


def get_task_proxy(tpu_name=None, uuid_var=None):
    machine = get_local_machine_name()
    return TaskProxy("gosford.cs.umass.edu", 8000, machine, tpu_name, uuid_var)


def get_task_manager_proxy():
    return TaskManagerProxy("gosford.cs.umass.edu", 8000)


def assign_tpu(wait=True):
    print("Auto assign TPU")
    machine = get_local_machine_name()
    if machine == "lesterny":
        condition = "v2"
    elif machine == "instance-3":
        condition = "v3"
    elif machine == "instance-4":
        condition = "v3"
    else:
        condition = None

    assigned_tpu = get_task_manager_proxy().get_tpu(condition)['tpu_name']

    sleep_time = 5
    while wait and assigned_tpu is None:
        time.sleep(sleep_time)
        if sleep_time < 300:
            sleep_time += 10
        assigned_tpu = get_task_manager_proxy().get_tpu(condition)['tpu_name']
    print("Assigned tpu : ", assigned_tpu)
    return assigned_tpu



if __name__ == "__main__":
    import numpy
    arr = numpy.array([0.1,0.242])
    value = arr[0]

    TaskManagerProxy("gosford.cs.umass.edu", 8000).get_num_active_jobs("lesterny")