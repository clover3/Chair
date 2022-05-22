from taskman_client.task_proxy import get_local_machine_name, TaskProxy


def flag_to_run_name(flags):
    if flags.run_name is not None:
        return flags.run_name
    else:
        return flags.output_dir.split("/")[-1]


def get_hp_str_from_flag(flags):
    log_key_flags = ["init_checkpoint", "input_files", "output_dir"]
    s = ""
    for key in log_key_flags:
        value = getattr(flags, key)
        s += "{}:\t{}\n".format(key, value)
    return s


def report_run3(func):
    def func_wrapper(args):
        flags = args
        run_name = flag_to_run_name(flags)
        machine = get_local_machine_name()
        task_proxy = TaskProxy("gosford.cs.umass.edu", 8000, machine, flags.tpu_name, None)
        flags_str = get_hp_str_from_flag(flags)
        if flags.use_tpu and flags.tpu_name is None:
            task_proxy.task_pending(run_name, flags_str)
            print("Requesting TPU...")
            flags.tpu_name = task_proxy.request_tpu(run_name)
            task_proxy.tpu_name = flags.tpu_name

        job_id = flags.job_id if flags.job_id >= 0 else None
        task_proxy.task_start(run_name, "", job_id)
        try:
            r = func(args)
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


