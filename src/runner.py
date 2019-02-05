import sys
import os

if __name__ == "__main__":
    st = int(sys.argv[1])
    interval = 10
    for i in range(st, st+ interval):
        sh_cmd = "PYTHONPATH=/mnt/nfs/work3/youngwookim/code/Chair/src /mnt/nfs/work3/youngwookim/miniconda3/envs/chair/bin/python3.6m data_generator/adhoc/merge_sample_encoder.py encode {} &".format(i)
        os.system(sh_cmd)

