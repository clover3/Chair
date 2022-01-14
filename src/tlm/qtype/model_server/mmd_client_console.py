from bert_api.client_lib import BERTClient
from port_info import MMD_Z_PORT


def main():
    client = BERTClient("http://localhost", MMD_Z_PORT, 512)
    while True:
        sent1 = input("Query: ")
        sent2 = input("Document: ")
        ret = client.request_single(sent1, sent2)
        print((sent1, sent2))
        print(ret)


if __name__ == "__main__":
    main()
