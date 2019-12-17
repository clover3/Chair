import json
import os

from sydney_manager import ReadyMarkTaskManager

iteration_dir = "/mnt/scratch/youngwookim/data/tlm_simple"
if not os.path.exists("/mnt/scratch/youngwookim/"):
    iteration_dir = "/mnt/nfs/work3/youngwookim/data/tlm_simple"

def get_path(sub_dir_name, file_name):
    out_path = os.path.join(iteration_dir, sub_dir_name, file_name)
    dir_path = os.path.join(iteration_dir, sub_dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return out_path


def delete_bad_query(path):
    j = json.load(open(path))
    new_q = []
    for e in j["queries"]:
        if e["text"] == "#combine()":
            pass
        else:
            new_q.append(e)

    new_j = {"queries":new_q}

    json.dump(new_j, open(path, "w"), indent=True)


def run_galago(job_idx):
    cmd = "galago threaded-batch-search --index=/mnt/nfs/work3/youngwookim/data/tlm/index/wiki_heads" \
          " --showNoResults --requested=30 {} > {}"
    input_path = get_path("query", "{}".format(job_idx))
    output_path = get_path("q_res", "{}.txt".format(job_idx))
    delete_bad_query(input_path)
    cmd = cmd.format(input_path, output_path)
    cmd = cmd
    print(cmd)
    os.system(cmd)

def galago_runner():
    iter = 1
    print("galago_runner")
    ready_sig = os.path.join(iteration_dir, "query", "{}")
    mark_path = os.path.join(iteration_dir, "mark", "galago_{}".format(iter))
    max_job = 1000

    mtm = ReadyMarkTaskManager(max_job, ready_sig, mark_path)
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        run_galago(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def etst():
    p = "/mnt/nfs/work3/youngwookim/data/tlm_simple/query/3"
    delete_bad_query(p)

def main():
    galago_runner()

if __name__ == "__main__":
    main()

