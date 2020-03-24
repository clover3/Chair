import os
import pickle


def split(disk_no):
    file_path = "/mnt/nfs/collections/ClueWeb12/ClueWeb12-DocID-To-URL/ClueWeb12_Disk{}_DocID_To_URL.txt".format(disk_no)
    f = open(file_path, "r")
    st = len("clueweb12-")
    ed = len("clueweb12-0000tw-00")

    def save(dir_name, obj):
        save_dir_name = dir_name + ".warc.gz"

        dir_path = os.path.join("/mnt/nfs/work3/youngwookim/data/clueweb12_text/1_00", save_dir_name)
        if os.path.exists(dir_path):
            save_path = os.path.join(dir_path, "url_to_doc_id")
            pickle.dump(obj, open(save_path, "wb"))
    ##
    print("")
    pre_dir_name = ""
    url_to_doc_id = {}
    for line in f:
        idx = line.find(",")
        doc_id = line[:idx]
        url = line[idx+1:].strip()

        dir_name = doc_id[st:ed]
        if dir_name != pre_dir_name and url_to_doc_id:
            save(pre_dir_name, url_to_doc_id)
            url_to_doc_id = {}
            pre_dir_name = dir_name

        url_to_doc_id[url.strip()] = doc_id

    print("Done")
    if url_to_doc_id:
        save(pre_dir_name, url_to_doc_id)


if __name__ == "__main__":
    split(1)


