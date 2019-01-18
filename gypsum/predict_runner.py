import os
from os.path import dirname

project_root = os.path.abspath(dirname(dirname((os.path.abspath(__file__)))))
src_path = os.path.join(project_root, "src")
config_dir_path = os.path.join(src_path, "config")
meta_run_path = os.path.join(project_root, "gypsum", "meta_run.sh")
format_path = os.path.join(config_dir_path, "predict_adhoc_robust.py.format")
config_path = os.path.join(config_dir_path, "predict_adhoc_robust.py")


def run_job(i):
    content = open(format_path, "r").read()
    content = content.replace("${idx}", str(i))
    open(config_path, "w").write(content)
    os.system("sh " + meta_run_path)


for i in range(10):
    run_job(i)

