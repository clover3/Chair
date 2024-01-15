import traceback

import requests

from taskman_client.RESTProxy import RESTProxy
from taskman_client.host_defs import webtool_host, webtool_port
from taskman_client.task_proxy import get_local_machine_name


class SubJobContext:
    def __init__(self, job_name, job_id, max_job):
        self.sub_job_proxy: JobGroupProxy = JobGroupProxy(job_name, max_job)
        self.job_id = job_id
        self.server_active = False

    def __enter__(self):
        try:
            self.sub_job_proxy.sub_job_start(self.job_id)
            self.server_active = True
        except requests.exceptions.ConnectTimeout as e:
            print(e)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.server_active:
            return

        if exc_type is None:
            self.sub_job_proxy.sub_job_done(self.job_id)
        else:
            if exc_type == KeyboardInterrupt:
                msg = "KeyboardInterrupt\n"
            else:
                msg = "Exception\n"
            msg += str(exc_type) + "\n"

            tb_str = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
            msg += tb_str + "\n"
            self.sub_job_proxy.sub_job_error(self.job_id, msg)


class JobGroupProxy:
    def __init__(self, job_name, max_job):
        self.machine = get_local_machine_name()
        self.job_name = job_name
        self.max_job = max_job
        self.server_proxy = RESTProxy(webtool_host, webtool_port)

    def sub_job_start(self, job_id):
        update_type = "START"
        return self._update_sub_job_inner(job_id, update_type, "")

    def sub_job_error(self, job_id, error_msg=""):
        update_type = "ERROR"
        return self._update_sub_job_inner(job_id, update_type, error_msg)

    def sub_job_done(self, job_id, msg=""):
        update_type = "DONE"
        return self._update_sub_job_inner(job_id, update_type, msg)

    def _update_sub_job_inner(self, job_id, update_type, msg=""):
        data = {
            'job_name': self.job_name,
            'machine': self.machine,
            'update_type': update_type,
            'msg': msg,
            "max_job": self.max_job,
            "job_id": job_id,
        }
        return self.server_proxy.post("/task/sub_job_update", data)

    def sub_job_context(self, job_id):
        return SubJobContext(self, self.job_name, job_id, self.max_job)
