import os
from os.path import dirname
import time

project_root = os.path.abspath(dirname(dirname((os.path.abspath(__file__)))))
src_path = os.path.join(project_root, "src")
config_dir_path = os.path.join(src_path, "config")
meta_run_path = os.path.join(project_root, "gypsum", "meta_run.sh")
sh_format_path = os.path.join(project_root, "gypsum", "adhoc_main_arg.sh.format")
sh_path_prefix = os.path.join(project_root, "gypsum", "adhoc_main_arg.sh")

def run_job(i):
    content = open(sh_format_path, "r").read()
    content = content.replace("${arg}", str(i))
    sh_path = sh_path_prefix + "_{}".format(i)
    open(sh_path, "w").write(content)

    sh_cmd = "sbatch -p 1080ti-short --gres=gpu:1 " + sh_path
    os.system(sh_cmd)
    time.sleep(1)

# (3,9)
for i in range(29, 99):
    run_job(i)

