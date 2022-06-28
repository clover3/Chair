import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer_v2.custom_loop.per_task.nli_ts_helper import get_local_decision_nlits_core
import sys

def main():
    run_name = sys.argv[1]
    encoder_name = sys.argv[2]
    nlits = get_local_decision_nlits_core(run_name, encoder_name)
    return NotImplemented


if __name__ == "__main__":
    main()