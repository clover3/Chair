import os
import subprocess
from typing import Dict


def trec_eval_wrap(label_path, prediction_path, metric_opt=""):
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
    if os.name == 'nt':
        return "C:\\work\\Tool\\trec_eval\\trec_eval.exe"
    else:
        return "trec_eval"


def run_trec_eval_parse(
        prediction_path, qrel_path,
        metric_opt=""
) -> Dict[str, str]:
    stdout, _ = trec_eval_wrap(qrel_path, prediction_path, metric_opt)
    msg = stdout.decode("utf-8")
    ret = {}
    for line in msg.strip().split("\n"):
        metric, _s_all, number = line.split()
        ret[metric] = number
    return ret