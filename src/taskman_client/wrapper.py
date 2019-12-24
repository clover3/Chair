from taskman_client.task_proxy import get_task_proxy, get_task_manager_proxy
from tlm.benchmark.report import get_hp_str_from_flag
from tlm.training.train_flags import FLAGS


def flag_to_run_name(FLAGS):
    if FLAGS.run_name is not None:
        return FLAGS.run_name
    else:
        return FLAGS.output_dir.split("/")[-1]


def report_number(r):
    try:
        value = float(r[FLAGS.report_field])

        condition = None
        if FLAGS.report_condition:
            condition = FLAGS.report_condition

        proxy = get_task_manager_proxy()
        proxy.report_number(FLAGS.run_name, value, condition)
    except Exception as e:
        print(e)


def report_run(func):
    def func_wrapper(*args):
        task_proxy = get_task_proxy(FLAGS.tpu_name)
        run_name = flag_to_run_name(FLAGS)
        flags_str = get_hp_str_from_flag(FLAGS)
        task_proxy.task_start(run_name, flags_str)
        try:
            r = func(*args)
            msg = "{}\n".format(r) + flags_str

            if FLAGS.report_field:
                report_number(r)

            task_proxy.task_complete(run_name, str(msg))
        except Exception as e:
            task_proxy.task_interrupted(run_name, "Exception\n" + str(e))
            raise
        except KeyboardInterrupt as e:
            task_proxy.task_interrupted(run_name, "KeyboardInterrupt\n" + str(e))
            raise

        return r

    return func_wrapper


