import os
os.environ["LOGGER"] = "2"

from tlm.benchmark.nli import run_nli


if __name__ == "__main__":
    run_name = "nli"
    run_nli(run_name, "model.ckpt-75000")