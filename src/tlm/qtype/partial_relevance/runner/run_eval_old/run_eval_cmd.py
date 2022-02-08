import sys

# Runs eval for Related against full query
from tlm.qtype.partial_relevance.runner.run_eval_old.run_partial_related_full_eval import run_eval


def main():
    dataset = sys.argv[1]
    method = sys.argv[2]
    policy_name = sys.argv[3]
    if len(sys.argv) > 4:
        model_interface = sys.argv[4]
    else:
        model_interface = "localhost"

    run_eval(dataset, method, policy_name, model_interface)


if __name__ == "__main__":
    main()
