from data_generator.NLI.nli import DataLoader


def fun():
    sequence_length = 400
    data_loader = DataLoader(sequence_length, "bert_voca.txt", True, True)
    for e in data_loader.get_raw_example(data_loader.train_file, "telephone"):
        s1, s2, l = e
        print(s1)


fun()