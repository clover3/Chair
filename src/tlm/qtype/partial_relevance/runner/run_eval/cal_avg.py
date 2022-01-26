import sys

from tlm.qtype.partial_relevance.calc_avg import print_avg


def run_by_argv():
    run_name = sys.argv[1]
    print_avg(run_name)



def main():
    run_name = sys.argv[1]
    print_avg(run_name)


if __name__ == "__main__":
    main()