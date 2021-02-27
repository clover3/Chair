import os
import pickle
from typing import List, Dict, Tuple

# /mnt/nfs/collections/ClueWeb09/corpus-spam60/ClueWeb09_English_7/en0093/spam60.clue.0000.0023.trecweb.gz
# clueweb09-en0093-65-44528
from base_type import FilePath
from clueweb.corpus_reading.trec_gz_reader import iter_docs
from cpath import at_output_dir
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from misc_lib import readlines_strip, group_by, get_dir_files, TimeEstimator, get_first


# /mnt/nfs/collections/ClueWeb09/corpus-spam60/ClueWeb09_English_1/en0001/spam60.clue.0000.0000.trecweb.gz
# clueweb09-en0001-00-05467


def get_clue09_subdir_mapping():
    clue09_subdirs: Dict[str, List[str]] = {
    "ClueWeb09_English_1": ["en0000", "en0001", "en0002", "en0003", "en0004", "en0005", "en0006", "en0007", "en0008", "en0009", "en0010", "en0011", "_enwp00", "_enwp01", "_enwp02", "_enwp03", ]
    ,"ClueWeb09_English_10":["en0124", "en0125", "en0126", "en0127", "en0128", "en0129", "en0130", "en0131", "en0132", "en0133", ]
    ,"ClueWeb09_English_2": ["en0012", "en0013", "en0014", "en0015", "en0016", "en0017", "en0018", "en0019", "en0020", "en0021", "en0022", "en0023", "en0024", "en0025", "en0026", ]
    ,"ClueWeb09_English_3": ["en0027", "en0028", "en0029", "en0030", "en0031", "en0032", "en0033", "en0034", "en0035", "en0036", "en0037", "en0038", "en0039", "en0040", ]
    ,"ClueWeb09_English_4": ["en0041", "en0042", "en0043", "en0044", "en0045", "en0046", "en0047", "en0048", "en0049", "en0050", "en0051", "en0052", "en0053", "en0054", ]
    ,"ClueWeb09_English_5": ["en0055", "en0056", "en0057", "en0058", "en0059", "en0060", "en0061", "en0062", "en0063", "en0064", "en0065", "en0066", "en0067", "en0068", ]
    ,"ClueWeb09_English_6": ["en0069", "en0070", "en0071", "en0072", "en0073", "en0074", "en0075", "en0076", "en0077", "en0078", "en0079", "en0080", "en0081", "en0082", ]
    ,"ClueWeb09_English_7": ["en0083", "en0084", "en0085", "en0086", "en0087", "en0088", "en0089", "en0090", "en0091", "en0092", "en0093", "en0094", "en0095", "en0096", ]
    ,"ClueWeb09_English_8": ["en0097", "en0098", "en0099", "en0100", "en0101", "en0102", "en0103", "en0104", "en0105", "en0106", "en0107", "en0108", "en0109", ]
    ,"ClueWeb09_English_9": ["en0110", "en0111", "en0112", "en0113", "en0114", "en0115", "en0116", "en0117", "en0118", "en0119", "en0120", "en0121", "en0122", "en0123", ]
    }

    d = {}
    for parent, children in clue09_subdirs.items():
        for child in children:
            if child[0] == "_":
                child = child[1:]
            d[child] = parent
    return d


group_to_parent = get_clue09_subdir_mapping()


def group_name_to_subdir_path(group_name):
    parent = group_to_parent[group_name]

    if group_name.startswith("enwp"):
        group_name = "_" + group_name
    return os.path.join(parent, group_name)


class Clueweb09DirectoryHelper:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def iter_gz_files_for_group(self, group_name):
        dir_to_iter = os.path.join(self.root_dir, group_name_to_subdir_path(group_name))
        file_list = get_dir_files(FilePath(dir_to_iter))
        file_list.sort()
        return file_list


def get_sydney_clueweb09_corpus_helper():
    return Clueweb09DirectoryHelper("/mnt/nfs/collections/ClueWeb09/corpus-spam60")


def get_doc_group(doc_id):##
    corpus_name, name1, name2, name3 = doc_id.split("-")
    return name1


class GetDocWorker:
    def __init__(self, todo, out_dir):
        self.out_dir = out_dir
        self.todo = todo

    def work(self, job_id):
        data_to_save = {}
        group_id, doc_ids = self.todo[job_id]
        cur_targets = set(doc_ids)
        dir_helper = get_sydney_clueweb09_corpus_helper()
        print(group_id, len(cur_targets))
        ticker = TimeEstimator(len(cur_targets))
        group_done = len(cur_targets) == 0
        for file_path in dir_helper.iter_gz_files_for_group(group_id):
            if group_done:
                break
            for doc_id, content in iter_docs(file_path):
                if doc_id in cur_targets:
                    data_to_save[doc_id] = content
                    cur_targets.remove(doc_id)
                    ticker.tick()

                if len(cur_targets) == 0:
                    group_done = True
                    break

        if cur_targets:
            print(len(cur_targets), "not found")

        pickle.dump(data_to_save, open(os.path.join(self.out_dir, str(job_id)), "wb"))


def main():
    doc_id_list = readlines_strip(at_output_dir("clueweb", "not_found.sort.txt"))
    grouped = group_by(doc_id_list, get_doc_group)

    todo: List[Tuple[str, List]] = list(grouped.items())
    todo.sort(key=get_first)
    num_jobs = len(grouped)

    def worker_factory(out_dir):
        return GetDocWorker(todo, out_dir)

    print("num jobs", num_jobs)
    runner = JobRunner(job_man_dir, num_jobs-1, "get_missing_clueweb09_docs", worker_factory)
    runner.start()


def num_files_to_touch():
    doc_id_list = readlines_strip(at_output_dir("clueweb", "not_found.sort.txt"))
    grouped = group_by(doc_id_list, get_doc_group)
    dir_helper = get_sydney_clueweb09_corpus_helper()
    for group_id, doc_ids in grouped.items():
        num_files = dir_helper.iter_gz_files_for_group(group_id)
        print(len(doc_ids), len(num_files))


if __name__ == "__main__":
    main()
