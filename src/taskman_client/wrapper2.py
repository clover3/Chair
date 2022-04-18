from taskman_client.task_proxy import get_task_proxy


def report_run_named(run_name):
    def report_run(func):
        def func_wrapper(*args):
            task_proxy = get_task_proxy()
            msg = ""
            task_proxy.task_start(run_name, msg)
            try:
                r = func(*args)
                print("Run completed")
                msg = "{}\n".format(r)
                print("Now reporting task : ", run_name)
                task_proxy.task_complete(run_name, str(msg))
                print("Done")
            except Exception as e:
                print("Reporting Exception ")
                task_proxy.task_interrupted(run_name, "Exception\n" + str(e))
                raise
            except KeyboardInterrupt as e:
                print("Reporting Interrupts ")
                task_proxy.task_interrupted(run_name, "KeyboardInterrupt\n" + str(e))
                raise

            return r

        return func_wrapper
    return report_run