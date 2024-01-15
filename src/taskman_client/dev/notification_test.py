from taskman_client.task_proxy import get_task_manager_proxy

proxy = get_task_manager_proxy()
proxy.make_success_notification()