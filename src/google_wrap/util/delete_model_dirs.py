import os

from google.cloud import storage

from google_wrap.get_storage_name import get_storage_name


def main():
    os.environ["storage_name"] = "clover_eu4"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/youngwookim/clovertpu-483d14c880bf.json"
    root_path = "training/model/"

    dir_list = open("/tmp/dir_to_del", "r").readlines()
    client = storage.Client()

    for dir_name in dir_list:
        file_path = root_path + dir_name.strip() + "/"
        blobs = client.list_blobs(get_storage_name(), prefix=file_path)
        print(file_path)
        for b in blobs:
            # print(b.name)
            b.delete()



if __name__ == "__main__":
    main()