from taskman_client.task_proxy import get_task_manager_proxy

tmp = get_task_manager_proxy()

r = tmp.get_tpu("v4")
print(r)