import subprocess

from taskman_client.task_proxy import get_local_machine_name


def trec_eval_wrap(label_path, prediction_path, metric_opt):
    exe = get_trec_eval_path()
    cmd = [exe, ]
    if metric_opt:
        cmd.append("-m {}".format(metric_opt))

    cmd.append(label_path)
    cmd.append(prediction_path)
    cmd = " ".join(cmd)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    return p.communicate()


def get_trec_eval_path():
    if get_local_machine_name() == "GOSFORD":
        return "C:\\work\\Tool\\trec_eval\\trec_eval.exe"
    else:
        return "trec_eval"