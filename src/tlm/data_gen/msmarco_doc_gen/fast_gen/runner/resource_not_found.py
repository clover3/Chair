from dataset_specific.msmarco.common import train_query_group_len, load_query_group
from epath import job_man_dir
import os


def main():
    query_groups = load_query_group("train")
    path_format = os.path.join(job_man_dir, "seg_resource_train", "{}")
    not_found_list = []
    for job_id in range(train_query_group_len):
        qids = query_groups[job_id]
        for qid in qids:
            resource_path = path_format.format(qid)
            if not os.path.exists(resource_path):
                print(job_id, qid)
                not_found_list.append((job_id, qid))

    print("{} files not found".format(len(not_found_list)))


if __name__ == "__main__":
    main()