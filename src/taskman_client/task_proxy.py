import os
import uuid

from taskman_client.RESTProxy import RESTProxy


class TaskProxy(RESTProxy):
    def __init__(self, host, port):
        super(TaskProxy, self).__init__(host, port)

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

    def task_start(self, run_name, tpu_name, machine, msg):
        uuid_var = str(uuid.uuid1())
        update_type = "START"

        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_start(self, run_name, tpu_name, machine, msg):
        self.uuid_var = str(uuid.uuid1())
        update_type = "START"

        return self.task_update(run_name, self.uuid_var, tpu_name, machine, update_type, msg)

    def task_complete(self, run_name, tpu_name, machine, msg):
        update_type = "SUCCESSFUL_TERMINATE"
        return self.task_update(run_name, self.uuid_var, tpu_name, machine, update_type, msg)

    def task_interrupted(self, run_name, tpu_name, machine, msg):
        update_type = "ABNORMAL_TERMINATE"
        return self.task_update(run_name, self.uuid_var, tpu_name, machine, update_type, msg)



class TaskProxyLocal:
    def __init__(self, host, port, machine, tpu_name=None):
        self.proxy = TaskProxy(host, port)
        self.tpu_name = tpu_name
        self.machine = machine

    def task_start(self, run_name, msg=None):
        return self.proxy.task_start(run_name, self.tpu_name, self.machine, msg)

    def task_complete(self, run_name, msg=None):
        return self.proxy.task_complete(run_name, self.tpu_name, self.machine, msg)

    def task_interrupted(self, run_name, msg=None):
        return self.proxy.task_interrupted(run_name, self.tpu_name, self.machine, msg)


def get_local_machine_name():
    if os.name == "nt":
        return os.environ["COMPUTERNAME"]
    else:
        return os.uname()[1]


def get_task_proxy(tpu_name=None):
    machine = get_local_machine_name()
    return TaskProxyLocal("gosford.cs.umass.edu", 8000, machine, tpu_name)


if __name__ == "__main__":
    TaskProxy("localhost", 8000).task_start("test_run", "null", "gosford", "test Msg" )