import os
from datetime import datetime

from misc_lib import exist_or_mkdir
from path import output_path


def save_report(task, run_name, flags, avg_acc):
    file_name = "{}".format(run_name)
    p = os.path.join(output_path, "report", file_name)
    exist_or_mkdir(os.path.join(output_path, "report"))
    f = open(p, "w")
    time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    f.write("{}:\t{}\n".format(task, avg_acc))
    f.write("{}\n".format(time_str))

    s = get_hp_str_from_flag(flags)
    f.write(s)


def get_hp_str_from_flag(flags):
    log_flags = ["init_checkpoint", "input_file", "output_dir",
                 "max_seq_length", "learning_rate", "train_batch_size"]
    s = ""
    for key in log_flags:
        value = getattr(flags, key)
        s += "{}:\t{}\n".format(key, value)

    return s
