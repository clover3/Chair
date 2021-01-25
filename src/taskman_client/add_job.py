from taskman_client.task_executer import get_next_sh_path_and_job_id


def run_job(sh_format_path, arg_map):

    content = open(sh_format_path, "r").read()
    for key, arg_val in arg_map.items():
        content = content.replace(key, arg_val)

    sh_path, job_id = get_next_sh_path_and_job_id()
    if "--job_id=-1" not in content:
        print("WARNING: --job_id=-1 is not in the script")
    content = content.replace("--job_id=-1", "--job_id={}".format(job_id))
    f = open(sh_path, "w")
    f.write(content)
    f.close()
    return job_id


def ukp_add_task(model_name, step, iteration):
    d = {
        "$1": model_name,
        "$2": step,
        "$3": iteration
    }
    run_job("ukp_generic_repeat.sh", d)
