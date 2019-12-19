import sys

from google_wrap.gs_wrap import download_model_last_auto
from tlm.benchmark.nli import run_nli


def download_and_run_nli(run_name):
    run_name, step_name= download_model_last_auto(run_name)
    run_nli(run_name, step_name)


if __name__ == "__main__":
    download_and_run_nli(sys.argv[1])