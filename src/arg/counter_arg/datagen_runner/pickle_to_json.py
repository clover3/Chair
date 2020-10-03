import json
import os
import pickle


def main():
    job_dir= "D:\job_dir\\arg_bert"

    obj = pickle.load(open(os.path.join(job_dir, "validation.info.pickle"), "rb"))
    json.dump(obj, open(os.path.join(job_dir, "validation.info"), "w"))



if __name__ == "__main__":
    main()