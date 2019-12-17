import os
import uuid

from taskman_client.RESTProxy import RESTProxy


class TaskManagerProxy(RESTProxy):
    def __init__(self, host, port):
        super(TaskManagerProxy, self).__init__(host, port)

    def task_update(self, run_name, uuid, tpu_name, machine, update_type, msg):
        data = {
            'run_name': run_name,
            'uuid': uuid,
            'tpu_name': tpu_name,
            'machine': machine,
            'update_type': update_type,
            'msg': msg
        }
        return self.post("/task/update", data)

    def task_start(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "START"

        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_complete(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "SUCCESSFUL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_interrupted(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "ABNORMAL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def report_number(self, name, value):
        data = {
            'name': name,
            "number": value
        }
        return self.post("/experiment/update", data)


class TaskProxy:
    def __init__(self, host, port, machine, tpu_name=None, uuid_var=None):
        self.proxy = TaskManagerProxy(host, port)
        self.tpu_name = tpu_name
        self.machine = machine

        if uuid_var is None:
            self.uuid_var = str(uuid.uuid1())
        else:
            self.uuid_var = uuid_var

    def task_start(self, run_name, msg=None):
        return self.proxy.task_start(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_complete(self, run_name, msg=None):
        return self.proxy.task_complete(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_interrupted(self, run_name, msg=None):
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

if __name__ == "__main__":
    TaskManagerProxy("gosford.cs.umass.edu", 8000).report_number("test_number", "3.14159")