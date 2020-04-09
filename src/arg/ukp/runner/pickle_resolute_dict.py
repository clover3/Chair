

# resolute_pid_cid.py
import sys

from arg.pf_common.resolute_dp_id import pickle_resolute_dict

if __name__ =="__main__":
    pickle_resolute_dict(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
