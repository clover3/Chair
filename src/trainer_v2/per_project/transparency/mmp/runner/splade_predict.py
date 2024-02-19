import xmlrpc.client

from cpath import at_output_dir
from dataset_specific.msmarco.passage.processed_resource_loader import load_msmarco_sample_dev_as_pairs
from misc_lib import write_to_lines


def main():
    score_fn = get_local_xmlrpc_scorer_fn()
    qd_pairs = list(load_msmarco_sample_dev_as_pairs())
    scores = score_fn(qd_pairs)
    # pass
    save_path = at_output_dir("lines_scores", "splade_dev_sample.txt")
    write_to_lines(scores, save_path)


def get_local_xmlrpc_scorer_fn():
    server_addr = "localhost"
    port = 28122
    proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(server_addr, port))
    score_fn = proxy.predict
    return score_fn


if __name__ == "__main__":
    main()