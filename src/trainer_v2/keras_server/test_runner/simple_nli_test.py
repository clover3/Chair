from trainer_v2.keras_server.name_short_cuts import get_nli14_predictor


def main():
    sent1 = 'This site includes a list of all award winners and a searchable database of Government Executive articles.'
    sent2 = 'The articles are not searchable.'
    predictor = get_nli14_predictor()

    ret = predictor([(sent1, sent2)])[0]
    print(ret)


if __name__ == "__main__":
    main()