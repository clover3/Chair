import os
os.environ["LOGGER"] = "2"
import sys
from tlm.benchmark.nli import run_nli_w_path

if __name__ == "__main__":
    run_name = sys.argv[1]
    step_name = sys.argv[2]
    full_path = sys.argv[3]

    run_nli_w_path(run_name, step_name, full_path)