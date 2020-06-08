from bert_api.client_lib import BERTClient

PORT_UKP = 8123


def run_test():
    client = BERTClient("http://localhost", PORT_UKP, 300)

    r = client.request_single("gun control", "when states passed concealed carry laws during the 19 years we studied ( 1977 to 1995 ) , the number of multiple-victim public shootings declined by 84 % .")
    print(r)
    r = client.request_single("gun control", "Education Is The Answer More harsh gun control laws are not needed .")
    print(r)

    r = client.request_single("abortion", "if you support abortion you are supporting murder")
    print(r)
    r = client.request_single("abortion", "abortion is only way to save lives.")
    print(r)


if __name__ == "__main__":
    run_test()