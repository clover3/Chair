from taskman_client.task_executer import get_new_job_id, get_last_mark, get_sh_path_for_job_id


def grep(text, pattern):
    for line in text.split("\n"):
        if pattern in line:
            return line

def enum_queued_task():
    st = get_last_mark() + 1
    ed = get_new_job_id()

    for id_idx in range(st, ed):
        yield id_idx

def get_run_names_of_queued_task():
    for id_idx in enum_queued_task():
        sh_path = get_sh_path_for_job_id(id_idx)
        content = open(sh_path, "r").read()
        line = grep(content, "run_name")
        print(sh_path)
        print(line)


def show_queued_ukp_repeat_style_jobs():
    for id_idx in enum_queued_task():
        sh_path = get_sh_path_for_job_id(id_idx)
        content = open(sh_path, "r").read()

        var_names = ["start_model_name", "start_model_step", "repeat_idx"]

        info_d = {}
        for var_name in var_names:
            line = grep(content, var_name)
            if line:
                offset = len(var_name) + 1
                var_value = line[offset:].strip()
                info_d[var_name] = var_value

        print(info_d)




if __name__ == "__main__":
    show_queued_ukp_repeat_style_jobs()

