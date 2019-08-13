import os
import path

import pickle

working_path ="/mnt/nfs/work3/youngwookim/data/tlm_simple"

def display(inst):
    target_tokens, prev_seg, cur_seg, next_seg, prev_tokens, next_tokens, mask_indice, doc_id = inst
    m_indice = set(mask_indice)

    def pretty(t):
        if t[:2] == "##":
            return t[2:]
        else:
            return t

    out_str = ""

    for i, t in enumerate(target_tokens):

        skip_space = t[:2] == "##"
        if not skip_space:
            out_str += " "

        t = pretty(t)
        if i in m_indice:
            out_str += "({})".format(t)
        else:
            out_str += t
    print(out_str)
    print("---------")


def main():
    job_id = 0
    out_path = os.path.join(working_path, "problems", "{}".format(job_id))
    obj = pickle.load(open(out_path, "rb"))
    for inst in obj:
        display(inst)


if __name__ == "__main__":
    main()
