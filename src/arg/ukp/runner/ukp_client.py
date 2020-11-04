import sys

from bert_api.client_lib import BERTClient

PORT_UKP = 8123


def load_inputs():
    for line in open(sys.argv[1], "r"):
        topic, sentence = line.split("\t")
        yield topic, sentence


def run_test():
    client = BERTClient("http://localhost", PORT_UKP, 300)

    data = list(load_inputs())

    r = client.request_multiple(data)

    for (topic, sent), score in zip(data, r):
        print("[{}] {}".format(topic, sent))
        print(score)


if __name__ == "__main__":
    run_test()