import os

import cpath

run_dir = os.path.join(cpath.model_path, 'runs')

def find_model_name(dir_name):
    save_dir = os.path.join(run_dir, dir_name)
    model_id = None
    for (dirpath, dirnames, filenames) in os.walk(save_dir):
        for filename in filenames:
            if ".meta" in filename:
                print(filename)
                model_id = filename[:-5]

    print(model_id)
    assert model_id is not None
    return dir_name, model_id


def find_available(dir_name):
    dir_name, model_id = find_model_name(dir_name)
    return os.path.join(run_dir, dir_name, model_id)
