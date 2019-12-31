import logging
import time

from taskman_client.task import Task
from taskman_client.task_executer import Executer
from taskman_client.task_executer import logger


def test1():
    logger.setLevel(logging.DEBUG)

    logger.info("Test1")
    manager = Executer()
    logger.info("Test2")
    manager.run()
    logger.info("Test3")
    task = Task()
    logger.info("Test4")
    task.argument_dict[""] = "C:\work\Data\TaskExecuter"
    task.process_name = "notepad.exe"
    manager.add_task_to_schedule(task)


    time.sleep(100)



if __name__ == "__main__":
    test1()