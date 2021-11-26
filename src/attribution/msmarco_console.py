from bert_api.client_lib_msmarco import BERTClientMSMarco


def main():
    client = BERTClientMSMarco("http://localhost", 8123, 512)
    sent1 = input("Query: ")
    while True:
        sent2 = input("Document: ")
        ret = client.request_single(sent1, sent2)
        print((sent1, sent2))
        print(ret)


if __name__ == "__main__":
    main()
