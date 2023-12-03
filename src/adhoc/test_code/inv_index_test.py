import random
import xmlrpc.client

from adhoc.other.inv_index_server import PORT_INV_INDEX


class InvIndexReaderClient:
    def __init__(self, server="localhost", port=PORT_INV_INDEX):
        self.server = xmlrpc.client.ServerProxy(f'http://{server}:{port}')

    def get_postings(self, key):
        return self.server.get_postings(key)


def main():
    client = InvIndexReaderClient()
    ret = client.get_postings("book")
    print("retrieve {} items".format(len(ret)))
    print(ret[:10])


def sample_inv_index(inv_index):
    compress_rate = 0.1
    new_inv_index = {}
    for q_term, entries in inv_index:
        n_sample = int(len(entries) * compress_rate)
        random.sample(entries, n_sample)
        new_inv_index[q_term] = random.sample(entries, n_sample)

    return new_inv_index



if __name__ == "__main__":
    main()

