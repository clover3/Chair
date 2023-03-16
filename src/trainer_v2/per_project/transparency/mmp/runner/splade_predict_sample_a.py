import xmlrpc.client

from cpath import at_output_dir
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sample_a_as_pairs


def main():
    server_addr = "localhost"
    port = 28122
    proxy = xmlrpc.client.ServerProxy('http://{}:{}'.format(server_addr, port))
    itr = iter(load_msmarco_sample_a_as_pairs())
    save_path = at_output_dir("lines_scores", "splade_dev_sample_a.txt")
    f = open(save_path, "w")

    batch_size = 1000
    qd_list = []
    try:
        while True:
            qd = next(itr)
            qd_list.append(qd)
            if len(qd_list) == batch_size:
                print("send {} request".format(len(qd_list)))
                scores = proxy.predict(qd_list)

            # pass
                for item in scores:
                    f.write("{}\n".format(item))
                qd_list = []
    except StopIteration:
        pass

    if qd_list:
        scores = proxy.predict(qd_list)
        for item in scores:
            f.write("{}\n".format(item))


if __name__ == "__main__":
    main()