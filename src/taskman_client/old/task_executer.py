import logging
import os
import threading
import time

import psutil

from misc_lib import exist_or_mkdir
from taskman_client.old.task import Task, STATUS_CANCELLED, STATUS_COMPLETED, STATUS_RUNNING, STATUS_WAIT
from taskman_client.sync import JsonTiedList, JsonTiedDict

logger = logging.getLogger("TaskExecuter")
logger.addHandler(logging.StreamHandler())

logger2 = logging.getLogger("TaskExecuter_thread")
logger2.addHandler(logging.StreamHandler())

root_dir = "/home/lesterny/te"
root_dir = "C:\work\Data\TaskExecuter"


class ResourceList:
    def __init__(self, resource_log_path, resource_name_list):
        self.resource_name_list = resource_name_list
        self.used_resource = JsonTiedList(resource_log_path)
        self.available_resource = list([t for t in self.resource_name_list if t not in self.used_resource])
        self.lock = threading.Lock()

    def assign(self):
        resource_name = None
        self.lock.acquire()
        if self.available_resource:
            resource_name = self.available_resource[0]
            self.available_resource.remove(resource_name)
            self.used_resource.add(resource_name)
        self.lock.release()
        return resource_name

    def release(self, resource_name):
        self.lock.acquire()
        self.available_resource.append(resource_name)
        self.used_resource.remove(resource_name)
        self.lock.release()


class TaskList:
    def __init__(self, tied_json_path, task_info_dir):
        self.task_info_dir = task_info_dir
        self.task_id_list = JsonTiedList(tied_json_path)
        self.task_id_to_obj = {}
        for task_id in self.task_id_list.list:
            self.task_id_to_obj[task_id] = self.load_task(task_id)
        self.lock = threading.Lock()

    def __iter__(self):
        for task_id in self.task_id_list.list:
            yield self.task_id_to_obj[task_id]

    def remove(self, task_obj:Task):
        self.lock.acquire()
        self.task_id_list.list.remove(task_obj.task_id)
        self.task_id_list.save()
        self.lock.release()


    def add(self, task_obj):
        self.lock.acquire()
        self.task_id_to_obj[task_obj.task_id] = task_obj
        self.task_id_list.list.append(task_obj.task_id)
        self.task_id_list.save()
        self.lock.release()

    def load_task(self, task_id):
        info_path = self._get_task_info_path(task_id)
        return QueuedTask.from_task(Task.from_json_file(info_path), info_path)

    def _get_task_info_path(self, task_id):
        return os.path.join(self.task_info_dir, "{}.json".format(task_id))


class QueuedTask(Task):
    def __init__(self, save_path):
        super(QueuedTask, self).__init__()
        self.save_path = save_path

    @classmethod
    def from_task(cls, task_obj, save_path):
        obj = QueuedTask(save_path)
        for key, value in task_obj.__dict__.items():
            obj.__dict__[key] = value
        return obj

    def _write_task(self):
        with open(self.save_path, "w") as f:
            f.write(self.to_json_string())

    def set_status(self, status):
        self.status = status
        self._write_task()


class Executer:
    def __init__(self):
        print("AAA")
        logger.debug("Executer init")
        # save/log current jobs, so that it can restart.
        self.task_info_dir = os.path.join(root_dir, "task_info")
        self.root_info_dir = os.path.join(root_dir, "root_info")
        exist_or_mkdir(self.task_info_dir)
        exist_or_mkdir(self.root_info_dir)

        # load task info for all active / queued task
        self.active_task_list = TaskList(os.path.join(self.root_info_dir, "active_task.json"),
                                         self.task_info_dir
                                         )
        self.queued_task_list = TaskList(os.path.join(self.root_info_dir, "queued_task.json"),
                                         self.task_info_dir)
        self.info_dict = JsonTiedDict(os.path.join(self.root_info_dir, "info.json"))

        tpu_info_path = os.path.join(self.root_info_dir, "tpu_info.json")
        self.tpu_resource = ResourceList(tpu_info_path, ["v2-tf2", "v2-tf2-2"])
        self.current_task_handles = {} # task_id -> process object
        # task_id being in current_task_handle does NOT imply the task is active, we don't delete handles
        self.task_cache = {} # task_id -> TaskObj
        self._init_info()

    def _get_new_task_id(self):
        new_task_id = self.info_dict.last_task_id + 1
        self.info_dict.set("last_task_id", new_task_id)
        return new_task_id

    def _get_task_info_path(self, task_id):
        return os.path.join(self.task_info_dir, "{}.json".format(task_id))

    def run(self):
        # start _thread
        t = threading.Thread(target=self._thread)
        t.daemon = True
        t.start()

    def add_task_to_schedule(self, task):
        task.task_id = self._get_new_task_id()
        logger.debug("add_task_to_schedule() task_id={} proc_name={}".format(task.task_id, task.process_name))
        new_task = QueuedTask.from_task(task, self._get_task_info_path(task.task_id))
        new_task.set_status(STATUS_WAIT)
        self.queued_task_list.add(new_task)

    def remove_task(self, task_name):
        # Kill task if it is active
        task_obj = self._remove_task_from_active_list(task_name)

        # Remove from the list if not active
        if task_obj is None:
            task_obj = self._remove_task_from_queued_list(task_name)
        return task_obj

    def _remove_task_from_active_list(self, task_name):
        deleted_task_obj = None
        for task_obj in self.active_task_list:
            if task_obj.task_name == task_name:
                self._kill_task(task_obj)
                task_obj.set_status(STATUS_CANCELLED)
                deleted_task_obj = task_obj
                break
        self.active_task_list.remove(deleted_task_obj)
        return deleted_task_obj

    def _remove_task_from_queued_list(self, task_name):
        deleted_task_obj = -1
        for task_obj in self.queued_task_list:
            if task_obj.task_name == task_name:
                task_obj.set_status(STATUS_CANCELLED)
                deleted_task_obj = task_obj
                break

        self.queued_task_list.remove(deleted_task_obj)
        return deleted_task_obj

    def _kill_task(self, task_obj):
        task_id = task_obj.task_id
        p = self.current_task_handles[task_id]
        p.kill()

    def _init_info(self):
        print("Init Info")
        logger.info("Init_info")
        logger2.info("Init Info")

        # Init self.current_task_handles
        task_to_mark_complete = []
        for task_obj in self.active_task_list:
            print("ActiveTask : ", task_obj.task_id)
            try:
                print("Acquiring Handle")

                logger.debug("Acquiring Handle {}".format(task_obj.task_id))
                self.current_task_handles[task_obj.task_id] = psutil.Process(task_obj.pid)
                logger.debug("Find task, task_id={} pid={}".format(task_obj.task_id, task_obj.pid))
            except psutil.NoSuchProcess as e:
                task_to_mark_complete.append(task_obj)

        self._clean_up_completed_list(task_to_mark_complete)

    # tpu_name should be already acquired before this function
    # TODO handle std_output redirection
    def _execute(self, task: Task, tpu_name=None):
        if tpu_name is not None:
            task.update_argument({"tpu_name":tpu_name})
        p = psutil.Popen([task.process_name, task.get_param_str()], env=task.env, shell=True,
                         )
        task.pid = p.pid
        return p

    def _task_sanity_check(self, task: Task):
        # TODO : Check if related gs files are available
        #   TODO : Cache information about gs file information
        # TODO : Check if necessary parameters are set
        return True

    def _thread(self):
        #  1. Poll Current Task status : By handle
        # 3. If resource is available, execute next task
        logger.info("_thread")
        while True:
            self._check_active_tasks()
            self._launch_task_if_possible()
            time.sleep(1)

    def _check_active_tasks(self):
        logger.info("check_active_tasks")
        task_to_mark_complete = []
        for task_obj in self.active_task_list:
            task_process: psutil.Process = self.current_task_handles[task_obj.task_id]
            try:
                status = task_process.status()
                logger.info("Task {} active".format(task_obj.task_id))
            except psutil.NoSuchProcess as e:
                status = "dead"
                logger.info("Task {} dead".format(task_obj.task_id))

            if status == "running":
                pass
            elif status == "dead":
                task_to_mark_complete.append(task_obj)
        # TODO
        #  2. Check stdout/stderr to see if process crashed

        self._clean_up_completed_list(task_to_mark_complete)

    def _launch_task_if_possible(self):
        task_that_just_got_executed = []
        for task_obj in self.queued_task_list:
            is_ready = True
            tpu_name = None
            if task_obj.use_tpu:
                tpu_name = self.tpu_resource.assign()
                if tpu_name is None:
                    is_ready = False

            if not self._task_sanity_check(task_obj):
                is_ready = False

            if is_ready:
                p = self._execute(task_obj, tpu_name)
                task_obj.pid = p.pid
                self.current_task_handles[task_obj.task_id] = p
                task_that_just_got_executed.append(task_obj)
            else:
                # return resource
                if tpu_name is not None:
                    self.tpu_resource.release(tpu_name)
        for task_obj in task_that_just_got_executed:
            logger.debug("execute() task_id={} proc_name={}".format(task_obj.task_id, task_obj.process_name))
            assert task_obj.pid is not None
            self.queued_task_list.remove(task_obj)
            self.active_task_list.add(task_obj)
            task_obj.set_status(STATUS_RUNNING)

    def _clean_up_completed_list(self, task_to_mark_complete):
        for task_obj in task_to_mark_complete:
            self.active_task_list.remove(task_obj)
            self._clean_up_completed(task_obj)

    def _clean_up_completed(self, task):
        logger.debug("_clean_up_completed() task_id={} ".format(task.task_id))
        task.set_status(STATUS_COMPLETED)
        if task.use_tpu:
            self.tpu_resource.release(task.tpu_name)

class ServerListener:
    def __init__(self):
        NotImplemented



    def _thread(self):
        NotImplemented