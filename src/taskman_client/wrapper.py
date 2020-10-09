from taskman_client.task_proxy import get_task_proxy, get_task_manager_proxy, assign_tpu
from tlm.benchmark.report import get_hp_str_from_flag
from tlm.training.train_flags import FLAGS


def flag_to_run_name(FLAGS):
    if FLAGS.run_name is not None:
        return FLAGS.run_name
    else:
        return FLAGS.output_dir.split("/")[-1]


def report_number(r):
    try:
        field = FLAGS.report_field
        if "," in field:
            field_list = field.split(",")
        else:
            field_list = [field]

        for field in field_list:
            value = float(r[field])

            condition = None
            if FLAGS.report_condition:
                condition = FLAGS.report_condition

            proxy = get_task_manager_proxy()
            proxy.report_number(FLAGS.run_name, value, condition, field)
    except Exception as e:
        print(e)


def report_run(func):
    def func_wrapper(*args):
        task_proxy = get_task_proxy(FLAGS.tpu_name)
        run_name = flag_to_run_name(FLAGS)
        flags_str = get_hp_str_from_flag(FLAGS)

        if FLAGS.use_tpu and FLAGS.tpu_name is None:
            task_proxy.task_pending(run_name, flags_str)
            FLAGS.tpu_name = assign_tpu()
            task_proxy.tpu_name = FLAGS.tpu_name

        flags_str = get_hp_str_from_flag(FLAGS)

        job_id = FLAGS.job_id if FLAGS.job_id >= 0 else None
        task_proxy.task_start(run_name, flags_str, job_id)
        try:
            r = func(*args)
            print("Run completed")
            msg = "{}\n".format(r) + flags_str

            if FLAGS.report_field:
                report_number(r)

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


