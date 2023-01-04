import time

from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_predictor, get_pep_client


def main():
    sent1 = 'This site includes a list of all award winners and a searchable database of Government Executive articles.'
    sent2 = 'The articles are not searchable.'
    predictor = get_keras_nli_300_predictor()

    st = time.time()
    ret = predictor([(sent1, sent2)])[0]

    ed = time.time()
    print("For one item took {} ".format(ed - st))

    n = 16

    payload = [(sent1, sent2)] * n
    print(len(payload))
    st = time.time()
    ret = predictor(payload)
    ed = time.time()
    print("For {} item took {} ({} per item) ".format(n, ed - st, (ed - st)/ n))


def main():
    sent1 = 'This site includes a list of all award winners and a searchable database of Government Executive articles.'
    sent2 = 'The articles are not searchable.'
    predictor = get_pep_client()

    st = time.time()
    ret = predictor([(sent1, sent2)])[0]
    print(ret)


if __name__ == "__main__":
    main()