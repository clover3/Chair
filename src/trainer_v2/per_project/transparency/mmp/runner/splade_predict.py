import xmlrpc.client

from cpath import at_output_dir
from dataset_specific.msmarco.passage.processed_resource_loader import load_msmarco_sample_dev_as_pairs
from misc_lib import write_to_lines


def main():
    server_addr = "localhost"
    port = 28122
    proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(server_addr, port))
    qd_pairs = list(load_msmarco_sample_dev_as_pairs())
    scores = proxy.predict(qd_pairs)
    # pass
    save_path = at_output_dir("lines_scores", "splade_dev_sample.txt")
    write_to_lines(scores, save_path)



if __name__ == "__main__":
    main()